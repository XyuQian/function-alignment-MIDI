import torch
import os
import numpy as np
import argparse
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from shoelace.utils.trainer_utils import Trainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tqdm import tqdm
from shoelace.actual_shoelace.data_loader import ShoelaceDataset as Dataset
from shoelace.actual_shoelace.data_loader import collate_fn, worker_init_fn
from shoelace.actual_shoelace.shoelace import Shoelace as Model
from shoelace.actual_shoelace.midi_2_audio.config import MODEL_FACTORY

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_dataset(rid, duration, batch_size, validation=False):
    num_workers = 0
    dataset = Dataset(
        duration=duration,
        validation=validation,
        path_folder="data/formatted",
        rid=rid,
        num_workers=num_workers
    )

    # For single-GPU training, we simply use shuffle (for training) instead of a distributed sampler
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=True,
        drop_last=True
    )

    return dataset, dataloader

def move_to_device(batch, dev):
    if isinstance(batch, dict):
        return {k : move_to_device(batch[k], dev) for k in batch}
    return batch.to(dev)
    
    
def del_batch(batch):
    if isinstance(batch, dict):
        for k in batch:
            del_batch(batch[k])
    del batch

@torch.no_grad()
def evaluate(model, dataloader, e, i, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    logging.info("Starting evaluation...")
    dl = tqdm(dataloader, desc=f"Evaluate epoch {e} step {i}")
    for batch in dl:
        model.reset_cache()
        loss = model(move_to_device(batch, device))
        total_loss += loss.item()
        num_batches += 1
        del_batch(batch)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    logging.info(f"Evaluation complete. Average Loss: {avg_loss:.4f}")
    model.train()
    return avg_loss

@torch.no_grad()
def save_model(model, writer, eval_loss, mean_loss, model_dir, step, e, i, min_loss):

    writer.add_scalar('train/mean_loss', mean_loss, step)
    model.save_weights(os.path.join(model_dir, f"latest_{e}_{i}"))
    if eval_loss < min_loss:
        min_loss = eval_loss
        model.save_weights(os.path.join(model_dir, f"best"))
        logging.info(f"Best checkpoint Epoch {e}, Step {i}: min Loss: {min_loss:.4f}")

    writer.add_scalar("eval/loss", eval_loss, step)
    logging.info(f"Epoch {e}, Step {i}: Eval Loss: {eval_loss:.4f}")
        
    return min_loss


def train(model, dataset, dataloader, duration, device, model_dir, learning_rate, epochs):
    num_steps = len(dataloader)
    rng = np.random.RandomState(456)
    writer = SummaryWriter(model_dir, flush_secs=20)
    trainer = Trainer(params=model.parameters(), lr=learning_rate, num_epochs=epochs, num_steps=num_steps)
    model = model.to(device)
    step = 0

    # Load validation dataset
    _, val_dataloader = get_dataset(rid=0, duration=duration, batch_size=16, validation=True)
    logging.info(f"Training started for {epochs} epochs.")

    min_loss = float('inf')

    eval_loss = evaluate(model, val_dataloader, 0, "end", device)
    min_loss = save_model(model, writer, eval_loss, 0, model_dir, step, 0, 0, min_loss)

    for e in range(epochs):
        mean_loss = 0
        n_element = 0
        model.train()

        dl = tqdm(dataloader, desc=f"Epoch {e}")
        # Reset random seed for dataset if applicable
        r = rng.randint(0, 912)
        dataset.reset_random_seed(r, e)

        logging.info(f"Epoch {e} started.")
        for i, batch in enumerate(dl):
            loss = model(move_to_device(batch, device))
            grad, lr = trainer.step(loss, model.parameters())
            step += 1

            writer.add_scalar("loss", loss.item(), step)
            writer.add_scalar("grad", grad, step)
            writer.add_scalar("lr", lr, step)
            n_element += 1
            mean_loss += loss.item()
            del_batch(batch)
            

            # Uncomment the lines below to perform periodic evaluation:
            # if i % 3000 == 0 and i > 0:
            #     eval_loss = evaluate(model, val_dataloader, e, i, device)
            #     min_loss = save_model(model, writer, eval_loss, mean_loss / n_element,
            #                           model_dir, step, e, i, min_loss)

        logging.info(f"Epoch {e} finished. Saving model...")
        model.save_weights(os.path.join(model_dir, f"latest_{e}_end"))

        eval_loss = evaluate(model, val_dataloader, e, "end", device)
        min_loss = save_model(model, writer, eval_loss, mean_loss / n_element, model_dir, step, e, "end", min_loss)


def main(args):
    experiment_folder = args.experiment_folder
    experiment_name = args.exp_name
    mask_type= args.mask_type

    os.makedirs(experiment_folder, exist_ok=True)
    model_dir = os.path.join(experiment_folder, experiment_name)
    os.makedirs(model_dir, exist_ok=True)

    logging.info(f"Experiment {experiment_name} started in {experiment_folder}")

    model = Model(device=torch.device(device), 
                mask_type=mask_type, model_configs=MODEL_FACTORY, model_names=["AudioLM", "MIDILM"])
    dataset, dataloader = get_dataset(duration=args.duration, rid=0, batch_size=args.batch_size)
    train(model, dataset, dataloader, args.duration, device, model_dir,
          learning_rate=args.learning_rate,
          epochs=args.epoch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--experiment_folder', type=str, required=True)
    parser.add_argument('-e', '--epoch', type=int, default=50)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-lr', '--learning_rate', type=float, required=True)
    parser.add_argument('-s', '--duration', type=float, required=True)
    parser.add_argument('-p', '--exp_name', type=str, required=True)
    parser.add_argument('-m', '--mask_type', type=str, required=True)

    args = parser.parse_args()

    logging.info("Starting script...")
    main(args)
