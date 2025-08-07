import torch
import os
import numpy as np
import argparse
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from tqdm import tqdm
from shoelace.actual_shoelace.midi_data_loader_new import PairedMIDIDataset, PairedMIDIDatasetSanity
from shoelace.actual_shoelace.midi_data_loader_new import collate_fn, worker_init_fn
from shoelace.actual_shoelace.midi_shoelace import Shoelace
from shoelace.actual_shoelace.midi_config import MODEL_FACTORY, TASKS, MODEL_MAPPING

# os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_dataset(rid, batch_size, task_type, num_workers, validation=False, rank=0, world_size=1):
    dataset = PairedMIDIDataset(
        validation=validation,
        path_folder="data/formatted/ASAP",
        rid=rid,
        task_type=task_type,
        num_workers=num_workers
    )

    sampler = None
    shuffle = True
    if world_size > 1: # Use DistributedSampler for DDP
        # For single-node, local_rank is often used as the device ID
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        shuffle = False # Sampler handles shuffling, so set DataLoader shuffle to False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle, # Set to False if using sampler
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True, # Enable memory pinning for faster CPU-GPU data transfer [1, 2]
        drop_last=True,
        sampler=sampler # Pass the sampler here
    )

    return dataset, dataloader


def get_sanity_dataset(rid, batch_size, task_type, modality, validation=False, rank=0, world_size=1):
    num_workers = 0 # Sanity check might not need multiple workers
    dataset = PairedMIDIDatasetSanity(
        validation=validation,
        path_folder="data/formatted/ASAP",
        rid=rid,
        task_type=task_type,
        num_workers=num_workers,
        modality=modality
    )

    sampler = None
    shuffle = True
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=True)
        shuffle = False

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        drop_last=True,
        sampler=sampler
    )

    return dataset, dataloader


def del_batch(batch):
    if isinstance(batch, dict):
        for k in batch:
            del_batch(batch[k])
    del batch


def move_to_device(batch, dev):
    if isinstance(batch, list):
        return batch
    if isinstance(batch, dict):
        return {k : move_to_device(batch[k], dev) for k in batch}
    return batch.to(dev)


@torch.no_grad()
def evaluate(model, dataloader, e, i, device, rank=0):
    model.eval()
    total_loss = 0
    num_batches = 0

    if rank == 0: # Only log on rank 0
        logging.info("Starting evaluation...")
    
    for batch in tqdm(dataloader, desc=f"Evaluate epoch {e} step {i}", disable=(rank!= 0)): # Disable tqdm for non-rank 0
        model.module.reset_cache()
        batch = move_to_device(batch, device)
        loss_dict = model(batch)
        loss = sum([loss_dict[k] for k in loss_dict])
        total_loss += loss.item()
        num_batches += 1
        del_batch(batch)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    
    if rank == 0: # Only log on rank 0
        logging.info(f"Evaluation complete. Average Loss: {avg_loss:.4f}")
    
    model.train()
    return avg_loss


@torch.no_grad()
def save_model(model, writer, eval_loss, mean_loss, model_dir, step, e, i, min_loss, suffix, rank=0):
    if rank == 0: # Only save/log on rank 0
        writer.add_scalar('train/mean_loss', mean_loss, step)
        # When using DDP, model.module refers to the original model
        model_to_save = model.module if isinstance(model, DDP) else model
        model_to_save.save_weights(os.path.join(model_dir, f"latest_{e}_{i}_{suffix}"))
        if eval_loss < min_loss:
            min_loss = eval_loss
            model_to_save.save_weights(os.path.join(model_dir, f"best"))
            logging.info(f"Best checkpoint Epoch {e}, Step {i}: min Loss: {min_loss:.4f}")
        
        writer.add_scalar("eval/loss", eval_loss, step)
        logging.info(f"Epoch {e}, Step {i}: Eval Loss: {eval_loss:.4f}")

    return min_loss


def train(model, dataloader, val_dataloader, device, model_dir, learning_rate, epochs, suffix, rank, world_size):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
    scaler = torch.amp.GradScaler('cuda')
    
    writer = None
    if rank == 0: # Only initialize SummaryWriter on rank 0
        writer = SummaryWriter(model_dir, flush_secs=20)

    step = 0
    min_loss = float("inf")

    # Initial evaluation and saving only on rank 0
    eval_loss = evaluate(model, val_dataloader, 0, step, device, rank)
    if rank == 0:
        min_loss = save_model(model, writer, eval_loss, 0, model_dir, step, 0, 0, min_loss, suffix, rank)
    
    # Ensure all processes wait for rank 0 to finish initial evaluation/saving
    if world_size > 1:
        dist.barrier()

    for epoch in range(epochs):
        # Set epoch for DistributedSampler if using it
        if world_size > 1 and hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(epoch)

        model.train()
        total_loss = 0
        n_element = 0 

        if rank == 0: # Only log gate values on rank 0
            for name, config in model.module.model_dict.items():
                for cross_attn_layer in config["adapter"].cross_attn:
                    gate = cross_attn_layer.gates.item()
                    logging.info(f"Layer {name} gate: {gate:.4f}")
                    writer.add_scalar(f"gate/{name}", gate, step)

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}", disable=(rank!= 0)): # Disable tqdm for non-rank 0 processes
            batch = move_to_device(batch, device)
            optimizer.zero_grad()

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss_dict = model(batch)
                loss = sum([loss_dict[k] for k in loss_dict])
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # loss_dict = model(batch)
            # loss = sum([loss_dict[k] for k in loss_dict])
            # loss.backward()
            # optimizer.step()

            step += 1
            n_element += 1

            if rank == 0: # Only log loss on rank 0
                for k in loss_dict:
                    writer.add_scalar(f"loss_{k}", loss_dict[k].item(), step)

            total_loss += loss.item()
            del_batch(batch)

        # Aggregate total_loss across all processes for accurate average loss calculation
        # Create a tensor for total_loss and reduce it
        if world_size > 1:
            # Convert total_loss to a tensor and move to device for reduction
            total_loss_tensor = torch.tensor(total_loss).to(device)
            dist.reduce(total_loss_tensor, dst=0, op=dist.ReduceOp.SUM)
            total_loss = total_loss_tensor.item() # Get the reduced value on rank 0

            # Sum the number of batches across all processes for correct average
            num_batches_this_epoch = torch.tensor(len(dataloader)).to(device)
            dist.reduce(num_batches_this_epoch, dst=0, op=dist.ReduceOp.SUM)
            total_batches_across_world = num_batches_this_epoch.item()
        else:
            total_batches_across_world = len(dataloader)

        avg_loss = total_loss / total_batches_across_world if total_batches_across_world > 0 else 0
        
        if rank == 0: # Only log/print on rank 0
            if epoch % 5 == 0:
                eval_loss = evaluate(model, val_dataloader, epoch, step, device, rank)
                writer.add_scalar("eval/loss", eval_loss, step)
                print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}, Eval Loss: {eval_loss:.4f}")
        scheduler.step()
    
    # Final evaluation and saving only on rank 0
    epoch += 1
    if rank == 0:
        eval_loss = evaluate(model, val_dataloader, epoch, "end", device, rank)
        writer.add_scalar("eval/loss", eval_loss, step)
        print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}, Eval Loss: {eval_loss:.4f}")
        print("Training complete.")
        min_loss = save_model(model, writer, eval_loss, avg_loss / n_element, model_dir, step, epoch, "end", min_loss, suffix, rank)
    
    # Clean up DDP processes
    if world_size > 1:
        dist.destroy_process_group()


def main(args):
    experiment_folder = args.experiment_folder
    experiment_name = args.exp_name
    model_dir = os.path.join(experiment_folder, experiment_name)

    # DDP setup
    if "LOCAL_RANK" in os.environ: # Check if launched by torchrun/DDP
        rank        = int(os.environ.get("RANK", 0))
        world_size  = int(os.environ.get("WORLD_SIZE", 1))
        
        # Initialize the process group
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(rank) # Set device for this process
        device = f"cuda:{rank}"

    else: # Single GPU/CPU training (for local testing without torchrun)
        rank = 0
        world_size = 1
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Only create directory and set up logging on rank 0 to avoid race conditions and cluttered logs
    if rank == 0:
        os.makedirs(experiment_folder, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(f"Experiment {experiment_name} started in {experiment_folder} on rank {rank}")
    else:
        # For non-rank 0 processes, set up a warning-level logger to suppress verbose output
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

    # Ensure all processes wait for rank 0 to create directories before proceeding
    if world_size > 1:
        dist.barrier()
    
    # Convert string to dict for mask_config
    mask_config = eval(args.mask_config) if isinstance(args.mask_config, str) else args.mask_config
    
    # Initialize the model
    model = Shoelace(
        device=torch.device(device),
        n_prompts=args.n_prompts,
        model_configs=MODEL_FACTORY,
        task_type=args.task_type,
        mask_config=mask_config
    ).to(device)

    # Wrap model with DDP *after* moving to device and *before* optimizer initialization
    if world_size > 1:
        model = DDP(model, device_ids=[rank])
    
    # Get datasets and dataloaders
    _, dataloader = get_dataset(rid=0, batch_size=args.batch_size, task_type=args.task_type, num_workers=args.num_workers, rank=rank, world_size=world_size)
    _, val_dataloader = get_dataset(rid=0, batch_size=args.batch_size, task_type=args.task_type, num_workers=args.num_workers, validation=True, rank=rank, world_size=world_size)

    train(
        model,
        dataloader,
        val_dataloader,
        device,
        model_dir,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        suffix=args.suffix,
        rank=rank,
        world_size=world_size
    )



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Arguments for distributed training (torchrun will pass these)
    parser.add_argument('--local_rank', type=int, help='local rank for distributed training')
    parser.add_argument('--rank', type=int, help='global rank for distributed training (set by torchrun)')
    parser.add_argument('--world_size', type=int, help='total number of distributed processes (set by torchrun)')
    parser.add_argument('--master_addr', type=str, help='master address for distributed training (set by torchrun)')
    parser.add_argument('--master_port', type=str, help='master port for distributed training (set by torchrun)')
    
    # Your existing arguments
    parser.add_argument('--batch_size', type=int, required=True, help='batch size PER GPU')
    parser.add_argument('--learning_rate', type=float, required=True, help='learning rate')
    parser.add_argument('--epochs', type=int, required=True, help='number of epochs')
    parser.add_argument('--suffix', type=str, required=True, help='suffix for model saving')
    parser.add_argument('--num_workers', type=int, required=True, help='number of workers for DataLoader')
    parser.add_argument('--experiment_folder', type=str, required=True, help='folder to save the experiment')
    parser.add_argument('--exp_name', type=str, required=True, help='name of the experiment')
    parser.add_argument('--task_type', type=str, required=True, help='task type for the model')
    parser.add_argument('--n_prompts', type=int, required=True, help='number of prompts for the model')
    parser.add_argument('--mask_config', type=str, required=True, help='mask configuration for the model')
    args = parser.parse_args()

    logging.info("Starting script...")
    main(args)