import torch
import json
from types import SimpleNamespace as Namespace

import os 
import sys 
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "hifi_gan"))
)

from models import Generator

def get_HIFI_GAN_MODEL(config_path, checkpoint_path, device):
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    with open(config_path, "r") as f:
        config_dict = json.load(f)
    config = Namespace(**config_dict)

    model = Generator(config).to(device)

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")

    # official checkpoints wrap generator under 'generator' key
    state_dict = checkpoint.get("generator", checkpoint)

    model.load_state_dict(state_dict)
    model.eval()
    model.remove_weight_norm()
    
    return model, config

@torch.no_grad()
def waveform_generation(model: Generator, log_mel: torch.Tensor) -> torch.Tensor:
    if log_mel.dim() == 2:
        log_mel = log_mel.unsqueeze(0)      # (1, n_mels, T)
    elif log_mel.dim() != 3:
        raise ValueError(f"log_mel must have shape (n_mels, T) or (batch, n_mels, T), got {log_mel.shape}")

    log_mel = log_mel.to(next(model.parameters()).device)
    audio = model(log_mel)                  # (batch, 1, T)
    audio = audio.squeeze(1)                # (batch, T)
    audio = torch.clamp(audio, -1.0, 1.0)

    if audio.size(0) == 1:
        return audio.squeeze(0)             # returns (T,) for single audio
    return audio
