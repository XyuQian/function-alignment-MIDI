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
from shoelace.actual_shoelace.shoelace import Yinyang as Model

device = "cuda"
SAMPLE_SEC = 16


def _get_free_port():
    import socketserver
    with socketserver.TCPServer(('localhost', 0), None) as s:
        return s.server_address[1]


def get_dataset(rid, is_mono, batch_size):
    num_workers = 0
    dataset = Dataset(path_folder="data/formatted/",
                      rid=rid,
                      is_mono=is_mono,
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
    model = Model(sec=SAMPLE_SEC, mode=args.mode)
    model = model.to(device)
    model.set_config(device)
    model = DDP(model, [replica_id])

    dataset, dataloader = get_dataset(rid=replica_id,
                                      batch_size=args.batch_size,
                                      is_mono=args.mono)

    train(replica_id, model, dataset, dataloader, device, model_dir,
          learning_rate=args.learning_rate,
          epochs=args.epoch)


def train(rank, model, dataset, dataloader, device, model_dir, learning_rate, epochs):
    # optimizer and lr scheduler
    num_steps = len(dataloader)
    rng = np.random.RandomState(345 + rank * 100)
    if rank == 0:
        writer = SummaryWriter(model_dir, flush_secs=20)

    trainer = Trainer(params=model.parameters(), lr=learning_rate, num_epochs=epochs, num_steps=num_steps)

    model = model.to(device)
    step = 0
    for e in range(0, epochs):
        mean_loss = 0
        n_element = 0
        model.train()

        dl = tqdm(dataloader, desc=f"Epoch {e}") if rank == 0 else dataloader
        r = rng.randint(0, 710)
        dataset.reset_random_seed(r, e)
        for i, batch in enumerate(dl):
            loss = 0
            loss_dict = model(batch)
            for n in loss_dict:
                loss += loss_dict[n][0] * loss_dict[n][1]
            grad, lr = trainer.step(loss, model.parameters())

            step += 1
            n_element += 1
            if rank == 0:
                for n in loss_dict:
                    writer.add_scalar(n, loss_dict[n][0].item(), step)
                writer.add_scalar("grad", grad, step)
                writer.add_scalar("lr", lr, step)

            mean_loss += (loss.item() / n_element)

            if rank == 0 and (e % 3 == 0 or e == epochs - 1) and i % 1600 == 0 and i > 0:
                with torch.no_grad():
                    model.module.save_weights(os.path.join(model_dir, f"latest_{e}_{i}"))

        if rank == 0 and e % 3 == 0 or e == epochs - 1:
            with torch.no_grad():
                model.module.save_weights(os.path.join(model_dir, f"latest_{e}_end"))
                writer.add_scalar('train/mean_loss', mean_loss, e)


def main(args):
    experiment_folder = args.experiment_folder
    experiment_name = args.exp_name

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
    parser.add_argument('-m', '--mode', type=str)
    parser.add_argument('-n', '--exp_name', type=str)
    parser.add_argument(
        '--mono',
        action='store_true',
    )

    args = parser.parse_args()
    main(args)
