import torch
import os
import numpy as np

import argparse
from torch.utils.tensorboard import SummaryWriter

from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from shoelace.utils.trainer_utils import Trainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tqdm import tqdm

from shoelace.actual_shoelace.shoelace_dataset import ShoelaceDataset as Dataset
from shoelace.actual_shoelace.shoelace_dataset import collate_fn, worker_init_fn
from shoelace.actual_shoelace.shoelace_2 import Yingyang as Model

device = "cuda"


def _get_free_port():
    import socketserver
    with socketserver.TCPServer(('localhost', 0), None) as s:
        return s.server_address[1]


def get_dataset(rid, is_mono, batch_size):
    num_workers = 0
    dataset = Dataset(is_mono=is_mono,
                      path_folder="data/formatted/groups",
                      rid=rid,
                      seg_sec=16,
                      num_workers=num_workers)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        sampler=DistributedSampler(dataset),
        pin_memory=True,
        drop_last=True)

    return dataset, dataloader


def train_dist(replica_id, replica_count, port, model_dir, args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(port)
    torch.distributed.init_process_group('nccl', rank=replica_id, world_size=replica_count)
    device = torch.device('cuda', replica_id)
    torch.cuda.set_device(device)
    model = Model(is_mono=args.mono)
    model = model.to(device)
    model.set_config(device)
    model.set_training(device)
    model = DDP(model, [replica_id])

    dataset, dataloader = get_dataset(rid=replica_id,
                                      is_mono=args.mono,
                                      batch_size=args.batch_size)

    train(replica_id, model, dataset, dataloader, device, model_dir,
          learning_rate=args.learning_rate,
          epochs=args.epoch)


def train(rank, model, dataset, dataloader, device, model_dir, learning_rate, epochs):
    # optimizer and lr scheduler
    num_steps = len(dataloader)
    # rng = np.random.RandomState(123 + rank * 100)
    rng = np.random.RandomState(345 + rank * 100)
    if rank == 0:
        writer = SummaryWriter(model_dir, flush_secs=20)

    trainer = Trainer(params=model.parameters(), lr=learning_rate, num_epochs=epochs, num_steps=num_steps)

    model = model.to(device)
    step = 0
    min_loss = 2333333
    for e in range(0, epochs):
        mean_loss = 0
        n_element = 0
        model.train()

        dl = tqdm(dataloader, desc=f"Epoch {e}") if rank == 0 else dataloader
        # r = rng.randint(0, 442)
        r = rng.randint(0, 710)
        dataset.reset_random_seed(r, e)
        for i, batch in enumerate(dl):

            audio_loss, midi_loss = model(**batch)
            loss = midi_loss + audio_loss
            grad, lr = trainer.step(loss, model.parameters())

            step += 1
            n_element += 1
            if rank == 0:
                writer.add_scalar("audio_loss", audio_loss.item(), step)
                writer.add_scalar("midi_loss", midi_loss.item(), step)
                writer.add_scalar("grad", grad, step)
                writer.add_scalar("lr", lr, step)

            mean_loss += loss.item()

            if rank == 0 and i > 0 and i % 3100 == 0:
                with torch.no_grad():
                    writer.add_scalar('train/mean_loss', mean_loss, step)
                    model.module.save_weights(os.path.join(model_dir, f"latest_{e}_{i}.pth"))

        mean_loss = mean_loss / n_element
        model.module.save_weights(os.path.join(model_dir, f"latest_{e}_end.pth"))


def main(args):
    experiment_folder = args.experiment_folder
    experiment_name = args.exp_name
    print("mono", args.mono)

    if not os.path.exists(experiment_folder):
        os.mkdir(experiment_folder)
    model_dir = os.path.join(experiment_folder, experiment_name)
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)
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
    parser.add_argument('--mono', action='store_true')
    parser.add_argument('-n', '--exp_name', type=str)

    args = parser.parse_args()
    main(args)
