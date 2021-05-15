import json 
import os
from argparse import ArgumentParser
from pathlib import Path
from multiprocessing import cpu_count
from uuid import uuid4
from tqdm import tqdm

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

from data import WavPreprocessNet
from model import load_pretrained_wav2vec

class PreprocessDataset(Dataset):
    def __init__(self, data_dir, wav_preprocess_net):
        self.wav_preprocess_net = wav_preprocess_net
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
        wav_processed = self.wav_preprocess_net(wav_tensor, sample_rate)
        return speaker_name, wav_processed

def main(
    data_dir: Path, 
    output_dir: Path, 
    wav2vec_path: Path,
    crop_len: int, 
):
    assert data_dir.is_dir()
    assert wav2vec_path.is_file()
    if output_dir.is_dir():
        os.system(f'rm -rf {output_dir}')
    feat_dir = output_dir / "feat"
    wav_dir = output_dir / "wav"
    feat_dir.mkdir(parents=True, exist_ok=True)
    wav_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_preprocess_net = WavPreprocessNet(crop_len)
    dataset = PreprocessDataset(data_dir, wav_preprocess_net)
    wav2vec = load_pretrained_wav2vec(wav2vec_path).to(device)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=cpu_count())
    infos = {
        "speakers": {speaker_name: [] for speaker_name in dataset.speakers},
    }

    for speaker_name, wav_tensor in tqdm(dataloader, ncols=0, desc="Preprocess"):
        speaker_name = speaker_name[0]
        wav_tensor = wav_tensor.to(device)
        feat = wav2vec.extract_features(wav_tensor, None)[0]
        feat = feat.detach().cpu().squeeze(0)
        feat_random_file_path = output_dir / "feat" /f"uttr-{uuid4().hex}.pt"
        torch.save(feat, feat_random_file_path)
        wav_tensor = wav_tensor.detach().cpu().squeeze(0)
        wav_random_file_path = output_dir / "wav" / f"uttr-{uuid4().hex}.pt"
        torch.save(wav_tensor, wav_random_file_path)
        infos["speakers"][speaker_name].append(
            {
                "wave_path": wav_random_file_path.name,
                "feature_path": feat_random_file_path.name,
            }
        )
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(infos, f, indent=2)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("data_dir", type=Path, help="Data directory.")
    parser.add_argument("output_dir", type=Path, help="Processing data output path.")
    parser.add_argument("wav2vec_path", type=Path, help="Wav2vec pretrained model path.")
    parser.add_argument("--crop_len", type=int, default=3)
    args = parser.parse_args()
    main(**vars(args))