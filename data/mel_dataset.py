"""Dataset for speaker embedding."""

import random
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio

class MelDataset(Dataset):
    def __init__(
        self,
        feat_dir: Union[str, Path],
        wav_dir: Union[str, Path],
        speaker_infos: dict,
    ):
        self.feat_dir = feat_dir
        self.wav_dir = wav_dir
        self.infos = []
        for uttr_infos in speaker_infos.values():
            for uttr_info in uttr_infos:
                self.infos.append((uttr_info['feature_path'],uttr_info['wave_path']))

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        feature_path, wave_path = self.infos[index]
        uttr = torch.load(Path(self.feat_dir, feature_path))
        wave = torch.load(Path(self.wav_dir, wave_path))
        return uttr, wave