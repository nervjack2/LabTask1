import json
from argparse import ArgumentParser
from pathlib import Path 
from multiprocessing import cpu_count

import torch 
from torch.utils.data import DataLoader 

from data import MelDataset

def main(
    data_dir: Path, 
    model_save_path: Path,
    n_utterances: int,
    seg_len: int,
    val_rate: float,
):
    # create data loader, iterator
    with open(Path(data_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    dataset = MelDataset(data_dir, metadata["speakers"], n_utterances, seg_len)
    n_valid = int(len(dataset)*val_rate)
    valset, trainset = dataset[:n_valid], dataset[n_valid:]
    valloader = DataLoader(
        valset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=cpu_count(),
        pin_memory=True
    )
    trainloader = DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=cpu_count(),
        pin_memory=True
    )

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=Path, help='Data directory path.')
    parser.add_argument('--model_save_path', type=Path, help='Path to save modle.')
    parser.add_argument('--n_utterances', type=int, default=500)
    parser.add_argument('--seg_len', type=int, default=150)
    parser.add_argument('--val_rate', type=float, default=0.1)
    args = parser.parse_args()
    main(**vars(args))