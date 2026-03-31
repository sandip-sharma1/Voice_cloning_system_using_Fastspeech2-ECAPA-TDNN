Voice Cloning System using FastSpeech2 & ECAPA-TDNN

Demo:

A complete voice cloning / multi-speaker text-to-speech (TTS) system that uses:

FastSpeech2 — Non-autoregressive TTS model for faster, high-quality speech generation.
ECAPA-TDNN — State-of-the-art speaker embedding model to capture speaker identity features.

This repository integrates TTS and speaker embedding models to clone a given speaker’s voice using a few reference utterances.

Features

--->Multi-speaker text-to-speech synthesis
--->Non-autoregressive TTS (fast inference & training)
--->Speaker identity capture via ECAPA-TDNN embeddings
--->Supports training, inference, evaluation, and similarity comparison
--->Includes utilities for preprocessing, alignment, and demo UI

Repository Contents

audio/ - audio preprocessing
config/ - Hyperparameters and config files
ecapa/ - ECAPA-TDNN speaker encoder
embeddings/ - Precomputed speaker embeddings
hifigan/ - Neural vocoder for waveform synthesis
lexicon/ - Phoneme lexicon
model/ - FastSpeech2 model definitions
similarity/ - Voice similarity metrics & utils
text/ - Text processing utilities
transformer/ - Transformer models & support
unseenembeddings/ - Unseen speaker embeddings support
utils/ - Shared utilities
train.py - Train TTS + speaker models
generate.py - Inference for speech synthesis
evaluate.py - Evaluation scripts
ui.py - Simple UI for demo/testing
requirements-*.txt - Requirements for Linux/Windows
README.md - This file

Installation

Clone the repository:

git clone https://github.com/sandip-sharma1/Voice_cloning_system_using_Fastspeech2-ECAPA-TDNN.git

cd Voice_cloning_system_using_Fastspeech2-ECAPA-TDNN

Setup Python environment:

python -m venv venv
source venv/bin/activate # Linux / macOS
venv\Scripts\activate # Windows

pip install -r requirements-linux.txt # Or requirements-windows.txt

Model Details

FastSpeech2 (Text-to-Speech):

--->Non-autoregressive architecture
--->Fast, parallel speech generation
--->Predicts mel-spectrogram from text input
--->Requires an aligned dataset for training (phoneme durations, etc.)

ECAPA-TDNN (Speaker Embeddings):

--->Extracts robust speaker identity features from audio
--->Helps clone voice characteristics across speakers
--->Achieves strong performance in speaker recognition tasks

Usage

steps to use this repo:
1) Clone the repository:

2) prepare_align
   run : python ./prepare_align.py ./config/LibriTTS/preprocess.yaml

3) Preprocess data
  run:python ./preprocess.py ./config/LibriTTS/preprocess.yaml

4) Train the model
  run:python ./train.py                        ^
    -p ./config/LibriTTS/preprocess.yaml ^
    -m ./config/LibriTTS/model.yaml      ^
    -t ./config/LibriTTS/train.yaml

5) inference
  run:python generate.py \
  --restore_step 500000 \
  --mode single \
  --text "Thank you" \
 --speaker_emb embeddings/LibriTTS/1.pt \
  -p config/LibriTTS/preprocess.yaml \
  -m config/LibriTTS/model.yaml \
  -t config/LibriTTS/train.yaml \
  --pitch_control 1.0 \
  --energy_control 1.0 \
  --duration_control 1.0


Modify config files for your dataset paths and hyperparameters.


Datasets

Supported datasets include:

LibriTTS — Clean, multi-speaker TTS dataset
Additional high-quality speech collections can be adapted

Ensure audio is sampled at consistent rate (e.g., 22050 Hz or 24 kHz) and aligned with text transcripts.

--->UI

Run a simple demo UI:

python ui.py

This provides options to load models, select reference voice, and synthesize speech.

Tips & Notes
--->Precompute speaker embeddings before synthesis.
--->Ensure phoneme alignment for training TTS model.
--->Use a vocoder like HiFi-GAN for waveform reconstruction.
--->Training on clean, high-quality data yields better voice cloning.

--->Requirements
See:

requirements-linux.txt
requirements-windows.txt

Install all dependencies before training or inference.


Acknowledgements
--->FastSpeech2 TTS research by Microsoft
--->ECAPA-TDNN speaker embedding model architecture
--->Open-source community for datasets & training utilities


