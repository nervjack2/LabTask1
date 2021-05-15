import json
from argparse import ArgumentParser
from pathlib import Path 
from multiprocessing import cpu_count

import torch
import torchaudio 
from torch.utils.data import DataLoader,random_split

from data import MelDataset
from model import VectorVocoder

def main(
    data_dir: Path, 
    model_save_path: Path,
    val_rate: float,
    batch_size: int, 
):
    device = torch.device("cuda" if torch.cuda.is_available else "cpu")
    # create data loader, iterator
    with open(Path(data_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    dataset = MelDataset(data_dir/"feat",data_dir/"wav", metadata["speakers"])
    n_valid = int(len(dataset)*val_rate)
    trainset, valset = random_split(dataset,[len(dataset)-n_valid, n_valid])
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

    model = VectorVocoder().to(device)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=Path, help='Data directory path.')
    parser.add_argument('--model_save_path', type=Path, help='Path to save modle.')
    parser.add_argument('--val_rate', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=4)
    args = parser.parse_args()
    main(**vars(args))