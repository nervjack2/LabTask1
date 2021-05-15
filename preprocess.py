import json 
from argparse import ArgumentParser
from pathlib import Path
from multiprocessing import cpu_count
from uuid import uuid4
from tqdm import tqdm

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from data import Wav2Mel

class PreprocessDataset(Dataset):
    def __init__(self, data_dir, wav2mel):
        self.wav2mel = wav2mel
        self.datainfo = []
        self.speakers = []
        for speaker_dir in data_dir.iterdir():
            self.speakers.append(speaker_dir.name)
            wav_dir = speaker_dir/'wav'
            for wav_file in wav_dir.iterdir():
                self.datainfo.append((speaker_dir.name, wav_file))
    def __len__(self):
        return len(self.datainfo)
    def __getitem__(self, idx):
        speaker_name, audio_path = self.datainfo[idx]
        wav_tensor, sample_rate = torchaudio.load(audio_path)
        mel_tensor = self.wav2mel(wav_tensor, sample_rate)
        return speaker_name, mel_tensor




def main(data_dir, output_dir):
    assert data_dir.is_dir() 
    output_dir.mkdir(parents=True, exist_ok=True)
    wav2mel = Wav2Mel()
    dataset = PreprocessDataset(data_dir, wav2mel)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=cpu_count())
    infos = {
        "n_mels": wav2mel.n_mels,
        "speakers": {speaker_name: [] for speaker_name in dataset.speakers},
    }

    for speaker_name, mel_tensor in tqdm(dataloader, ncols=0, desc="Preprocess"):
        speaker_name = speaker_name[0]
        mel_tensor = mel_tensor.squeeze(0)
        random_file_path = output_dir / f"uttr-{uuid4().hex}.pt"
        torch.save(mel_tensor, random_file_path)
        infos["speakers"][speaker_name].append(
            {
                "feature_path": random_file_path.name,
                "mel_len": len(mel_tensor),
            }
        )

    with open(output_dir / "metadata.json", "w") as f:
        json.dump(infos, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_dir", type=Path, help="Data directory.")
    parser.add_argument("output_dir", type=Path, help="Processing data output path.")
    args = parser.parse_args()
    main(**vars(args))