"""Wav2Mel for processing audio data."""

import torch
import torch.nn as nn
from torchaudio.sox_effects import apply_effects_tensor
from torchaudio.transforms import MelSpectrogram
from .utils import crop_segment

class WavPreprocessNet(nn.Module):
    """Transform audio file into mel spectrogram tensors."""

    def __init__(
        self,
        crop_len: int,
        sample_rate: int = 16000,
        norm_db: float = -3.0,
        sil_threshold: float = 1.0,
        sil_duration: float = 0.1,
    ):
        super().__init__()
        self.crop_len = crop_len
        self.sample_rate = sample_rate
        self.norm_db = norm_db
        self.sil_threshold = sil_threshold
        self.sil_duration = sil_duration
        self.sox_effects = SoxEffects(sample_rate, norm_db, sil_threshold, sil_duration)

    def forward(self, wav_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        wav_tensor = crop_segment(self.sox_effects(wav_tensor, sample_rate).squeeze(0), self.crop_len)
        return wav_tensor


class SoxEffects(nn.Module):
    """Transform waveform tensors."""

    def __init__(
        self,
        sample_rate: int,
        norm_db: float,
        sil_threshold: float,
        sil_duration: float,
    ):
        super().__init__()
        self.effects = [
            ["channels", "1"],  # convert to mono
            ["rate", f"{sample_rate}"],  # resample
            ["norm", f"{norm_db}"],  # normalize to -3 dB
            [
                "silence",
                "1",
                f"{sil_duration}",
                f"{sil_threshold}%",
                "-1",
                f"{sil_duration}",
                f"{sil_threshold}%",
            ],  # remove silence throughout the file
        ]

    def forward(self, wav_tensor: torch.Tensor, sample_rate: int) -> torch.Tensor:
        wav_tensor, _ = apply_effects_tensor(wav_tensor, sample_rate, self.effects)
        return wav_tensor
