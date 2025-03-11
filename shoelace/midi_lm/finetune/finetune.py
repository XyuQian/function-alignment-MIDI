import torch
import os
import numpy as np
import argparse
import logging
from torch.utils.tensorboard import SummaryWriter
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from shoelace.utils.trainer_utils import Trainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tqdm import tqdm
from shoelace.midi_lm.finetune.data_loader import MIDIDataset as Dataset
from shoelace.midi_lm.finetune.data_loader import collate_fn, worker_init_fn
from shoelace.midi_lm.finetune.midi_lm import MIDILMLora as Model

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

device = "cuda"


def _get_free_port():
    import socketserver
    with socketserver.TCPServer(('localhost', 0), None) as s:
        return s.server_address[1]


def get_dataset(rid, batch_size, validation=False):
    num_workers = 0
    dataset = Dataset(
        validation=validation,
        path_folder="data/formatted",
        rid=rid,
        num_workers=num_workers
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        sampler=DistributedSampler(dataset),
        pin_memory=True,
        drop_last=True
    )

    return dataset, dataloader


@torch.no_grad()
def evaluate(model, dataloader, e, i, device):
    model.eval()
    total_loss = 0
    num_batches = 0

    logging.info("Starting evaluation...")
    dl = tqdm(dataloader, desc=f"Evaluate epoch {e} step {i}")
    for batch in dl:
        loss = model(**batch)
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    logging.info(f"Evaluation complete. Average Loss: {avg_loss:.4f}")

    return avg_loss


def save_model(model, writer, eval_loss, mean_loss, model_dir, step, e, i, min_loss):
    with torch.no_grad():
        writer.add_scalar('train/mean_loss', mean_loss, step)
        model.module.save_weights(os.path.join(model_dir, f"latest_{e}_{i}"))
        if eval_loss < min_loss:
            min_loss = eval_loss
            model.module.save_weights(os.path.join(model_dir, f"best"))
            logging.info(f"Best checkpoint Epoch {e}, Step {i}: min Loss: {min_loss:.4f}")

        writer.add_scalar("eval/loss", eval_loss, step)
        logging.info(f"Epoch {e}, Step {i}: Eval Loss: {eval_loss:.4f}")
        model.train()
    return min_loss


def train(rank, model, dataset, dataloader, device, model_dir, learning_rate, epochs):
    num_steps = len(dataloader)
    rng = np.random.RandomState(456 + rank * 100)
    if rank == 0:
        writer = SummaryWriter(model_dir, flush_secs=20)

    trainer = Trainer(params=model.parameters(), lr=learning_rate, num_epochs=epochs, num_steps=num_steps)
    model = model.to(device)
    step = 0

    # Load validation dataset
    _, val_dataloader = get_dataset(rid=rank, batch_size=16, validation=True)

    logging.info(f"Training started for {epochs} epochs.")

    min_loss = 2333333

    eval_loss = evaluate(model, val_dataloader, 0, 0, device)
    if rank == 0:
        min_loss = save_model(model, writer, eval_loss, 0,
                              model_dir, step, 0, 0, min_loss)

    for e in range(epochs):
        mean_loss = 0
        n_element = 0
        model.train()

        dl = tqdm(dataloader, desc=f"Epoch {e}") if rank == 0 else dataloader
        r = rng.randint(0, 912)
        dataset.reset_random_seed(r, e)

        logging.info(f"Epoch {e} started.")

        for i, batch in enumerate(dl):
            loss = model(**batch)
            grad, lr = trainer.step(loss, model.parameters())

            step += 1

            if rank == 0:
                writer.add_scalar("loss", loss.item(), step)
                writer.add_scalar("grad", grad, step)
                writer.add_scalar("lr", lr, step)
                n_element += 1
                mean_loss += loss.item()

            # if (e == 0 or i > 0) and i % 3000 == 0:
            #     eval_loss = evaluate(model, val_dataloader, e, i, device)
            #
            # if rank == 0 and (e == 0 or i > 0) and i % 3000 == 0:
            #     min_loss = save_model(model, writer, eval_loss,
            #                           mean_loss / n_element,
            #                           model_dir, step, e, i, min_loss)

        logging.info(f"Epoch {e} finished. Saving model...")
        model.module.save_weights(os.path.join(model_dir, f"latest_{e}_end.pth"))

        eval_loss = evaluate(model, val_dataloader, e, "end", device)
        if rank == 0:
            min_loss = save_model(model, writer, eval_loss, mean_loss / n_element,
                                  model_dir, step, e, "end", min_loss)


def train_dist(replica_id, replica_count, port, model_dir, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)
    device = torch.device('cuda', replica_id)
    torch.cuda.set_device(device)

    logging.info(f"Initializing process group for replica {replica_id}/{replica_count}")

    model = Model(model_path="save_models/midi_lm_0311.pth")
    model = model.to(device)
    model = DDP(model, [replica_id])

    dataset, dataloader = get_dataset(rid=replica_id, batch_size=args.batch_size)

    logging.info(f"Starting training on replica {replica_id}")

    train(replica_id, model, dataset, dataloader, device, model_dir,
          learning_rate=args.learning_rate,
          epochs=args.epoch)


def main(args):
    experiment_folder = args.experiment_folder
    experiment_name = args.exp_name

    os.makedirs(experiment_folder, exist_ok=True)
    model_dir = os.path.join(experiment_folder, experiment_name)
    os.makedirs(model_dir, exist_ok=True)

    logging.info(f"Experiment {experiment_name} started in {experiment_folder}")

    world_size = args.world_size
    port = _get_free_port()
    spawn(train_dist, args=(world_size, port, model_dir, args), nprocs=world_size, join=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--experiment_folder', type=str)
    parser.add_argument('-e', '--epoch', type=int, default=50)
    parser.add_argument('-b', '--batch_size', type=int, default=64)
    parser.add_argument('-w', '--world_size', type=int, default=5)
    parser.add_argument('-lr', '--learning_rate', type=float)
    parser.add_argument('-p', '--exp_name', type=str)

    args = parser.parse_args()

    logging.info("Starting script...")
    main(args)
