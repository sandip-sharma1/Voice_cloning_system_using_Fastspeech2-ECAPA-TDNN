from utils import patch

import torch
from speechbrain.inference import SpeakerRecognition

def get_ECAPA_TDNN_MODEL(device, model_dir = "ecapa"):
    model = SpeakerRecognition.from_hparams(
        source=model_dir,
        run_opts = {"device": device},
        savedir=model_dir,
    )
    if model is None:
        raise RuntimeError("Failed to load ECAPA-TDNN model.")
    
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    return model

def speaker_embedding_extractor(model: SpeakerRecognition, waveform: torch.Tensor):
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    elif waveform.ndim != 2:
        raise ValueError(f"Expected waveform of shape [time] or [batch, time], got {waveform.shape}")
    
    device = next(model.parameters()).device
    waveform = waveform.float().to(device)

    with torch.no_grad():
        embeddings = model.encode_batch(waveform)       # [batch, 1, embedding_dim]
        embeddings = embeddings.squeeze(1)
        embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)      # L2 normalize

    return embeddings
