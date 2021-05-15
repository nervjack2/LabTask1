import random 
from torchaudio.sox_effects import apply_effects_tensor

def crop_segment(tensor, tgt_dur, sample_rate=16000):
    src_dur = len(tensor) / sample_rate
    random_shift = random.uniform(0, src_dur - tgt_dur)
    audio_tensor, _ = apply_effects_tensor(
        tensor.unsqueeze(0),
        sample_rate,
        [
            ["pad", f"{tgt_dur}", f"{tgt_dur}"],
            [
                "trim",
                f"{tgt_dur + random_shift}",
                f"{tgt_dur}",
            ],
        ],
    )
    return audio_tensor.squeeze(0)