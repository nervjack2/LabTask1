"""Dataset for speaker embedding."""

import random
from pathlib import Path
from typing import Union

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class MelDataset(Dataset):
    def __init__(
        self,
        data_dir: Union[str, Path],
        speaker_infos: dict,
        n_utterances: int,
        seg_len: int,
    ):
        self.data_dir = data_dir
        self.n_utterances = n_utterances
        self.seg_len = seg_len
        self.infos = []

        for uttr_infos in speaker_infos.values():
            for uttr_info in uttr_infos:
                if(uttr_info['mel_len'] > seg_len):
                    self.infos.append(uttr_info['feature_path'])

    def __len__(self):
        return len(self.infos)

    def __getitem__(self, index):
        feature_path = self.infos[index]
        uttr = torch.load(Path(self.data_dir, feature_path))
        left = random.randint(0, len(uttr) - self.seg_len)
        segments = uttr[left : left + self.seg_len, :]
        return segments