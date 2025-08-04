import torch
import os
import numpy as np
import argparse
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from shoelace.utils.trainer_utils import Trainer

from tqdm import tqdm
from shoelace.actual_shoelace.midi_data_loader import PairedMIDIDataset, PairedMIDIDatasetSanity
from shoelace.actual_shoelace.midi_data_loader import collate_fn, worker_init_fn
from shoelace.actual_shoelace.midi_shoelace import Shoelace
from midi_config import MODEL_FACTORY, TASKS, MODEL_MAPPING

device = "cuda" if torch.cuda.is_available() else "cpu"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def get_dataset(rid, batch_size, task_type, validation=False):
    num_workers = 0
    dataset = PairedMIDIDataset(
        validation=validation,
        path_folder="data/formatted/ASAP",
        rid=rid,
        task_type=task_type,
        num_workers=num_workers
    )

    # For single-GPU training, we simply use shuffle (for training) instead of a distributed sampler
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        drop_last=True
    )

    return dataset, dataloader


def get_sanity_dataset(rid, batch_size, task_type, modality, validation=False):
    num_workers = 0
    dataset = PairedMIDIDatasetSanity(
        validation=validation,
        path_folder="data/formatted/ASAP",
        rid=rid,
        task_type=task_type,
        num_workers=num_workers,
        modality=modality
    )

    # For single-GPU training, we simply use shuffle (for training) instead of a distributed sampler
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        drop_last=True
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
def evaluate(model, dataloader, e, i, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    logging.info("Starting evaluation...")
    for batch in tqdm(dataloader, desc=f"Evaluate epoch {e} step {i}"):
        model.reset_cache()
        batch = move_to_device(batch, device)
        loss_dict = model(batch)
        loss = sum([loss_dict[k] for k in loss_dict])
        total_loss += loss.item()
        num_batches += 1
        del_batch(batch)
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    logging.info(f"Evaluation complete. Average Loss: {avg_loss:.4f}")
    model.train()
    return avg_loss


@torch.no_grad()
def save_model(model, writer, eval_loss, mean_loss, model_dir, step, e, i, min_loss, suffix):
    writer.add_scalar('train/mean_loss', mean_loss, step)
    model.save_weights(os.path.join(model_dir, f"latest_{e}_{i}_{suffix}"))
    if eval_loss < min_loss:
        min_loss = eval_loss
        model.save_weights(os.path.join(model_dir, f"best"))
        logging.info(f"Best checkpoint Epoch {e}, Step {i}: min Loss: {min_loss:.4f}")
    
    writer.add_scalar("eval/loss", eval_loss, step)
    logging.info(f"Epoch {e}, Step {i}: Eval Loss: {eval_loss:.4f}")

    return min_loss


def train(model, dataloader, val_dataloader, device, model_dir, learning_rate, epochs, suffix):
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)
    writer = SummaryWriter(model_dir, flush_secs=20)

    step = 0
    min_loss = float("inf")

    eval_loss = evaluate(model, val_dataloader, 0, step, device)
    min_loss = save_model(model, writer, eval_loss, 0, model_dir, step, 0, 0, min_loss, suffix)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        n_element = 0

        for name, config in model.model_dict.items():
            for cross_attn_layer in config["adapter"].cross_attn:
                gate = cross_attn_layer.gates.item()
                logging.info(f"Layer {name} gate: {gate:.4f}")
                writer.add_scalar(f"gate/{name}", gate, step)

        for batch in tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}"):
            batch = move_to_device(batch, device)
            optimizer.zero_grad()
            loss_dict = model(batch)
            loss = sum([loss_dict[k] for k in loss_dict])
            loss.backward()
            optimizer.step()

            step += 1
            n_element += 1

            for k in loss_dict:
                writer.add_scalar(f"loss_{k}", loss_dict[k].item(), step)

            total_loss += loss.item()
            del_batch(batch)

        avg_loss = total_loss / len(dataloader)
        if epoch % 5 == 0:
            eval_loss = evaluate(model, val_dataloader, epoch, step, device)
            writer.add_scalar("eval/loss", eval_loss, step)
            print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}, Eval Loss: {eval_loss:.4f}")
        scheduler.step()
    epoch += 1
    eval_loss = evaluate(model, val_dataloader, epoch, "end", device)
    writer.add_scalar("eval/loss", eval_loss, step)
    print(f"Epoch {epoch}/{epochs}, Loss: {avg_loss:.4f}, Eval Loss: {eval_loss:.4f}")
    
    print("Training complete.")
    min_loss = save_model(model, writer, eval_loss, avg_loss / n_element, model_dir, step, epoch, "end", min_loss, suffix)


if __name__ == "__main__":
    experiment_folder = "exp"
    experiment_name = "midi_conversion"
    os.makedirs(experiment_folder, exist_ok=True)

    model_dir = os.path.join(experiment_folder, experiment_name)
    os.makedirs(model_dir, exist_ok=True)

    logging.info(f"Experiment {experiment_name} started in {experiment_folder}")

    model = Shoelace(
        device=torch.device(device),
        n_prompts=5,
        model_configs=MODEL_FACTORY,
        task_type="midi_conversion",
        mask_config={
            "ScoreLM": True,
            "PerformanceLM": False
        }
    ).to(device)

    dataset, dataloader = get_dataset(rid=0, batch_size=16, task_type="midi_conversion")
    _, val_dataloader = get_dataset(rid=0, batch_size=16, task_type="midi_conversion", validation=True)

    # For sanity check
    # dataset, dataloader = get_sanity_dataset(rid=0, batch_size=12, task_type="midi_conversion", modality="Score")
    # _, val_dataloader = get_sanity_dataset(rid=0, batch_size=12, task_type="midi_conversion", modality="Score", validation=True)

    # dataset, dataloader = get_sanity_dataset(rid=0, batch_size=16, task_type="midi_conversion", modality="Performance")
    # _, val_dataloader = get_sanity_dataset(rid=0, batch_size=16, task_type="midi_conversion", modality="Performance", validation=True)

    train(model, dataloader, val_dataloader, device, model_dir, learning_rate=5e-5, epochs=50, suffix="perf_2_score")

    # for name, param in model.named_parameters():
    #     if param.requires_grad:
    #         print(f"{name}: {param.size()}, requires_grad={param.requires_grad}")