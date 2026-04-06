Voice Cloning System using FastSpeech2 & ECAPA-TDNN

Demo on LinkedIn:https://www.linkedin.com/posts/sandip-sharma-0a7b98324_voicecloning-texttospeech-deeplearning-ugcPost-7446263735419297792-gEgC?utm_source=share&utm_medium=member_ios&rcm=ACoAAFILk38BbQv--mhT2lmnolKhkqoXrAor67k

A complete multi-speaker voice cloning and text-to-speech (TTS) system that leverages FastSpeech2 for high-quality, non-autoregressive speech synthesis and ECAPA-TDNN for robust speaker embeddings. This repository allows cloning a speaker’s voice using only a few reference utterances.

Features

Multi-speaker TTS: Synthesize speech in multiple voices.
Non-autoregressive TTS: Fast and parallel inference for high-quality speech.
Speaker identity capture: Use ECAPA-TDNN embeddings to preserve speaker characteristics.
End-to-end workflow: Supports data preprocessing, model training, inference, evaluation, and voice similarity analysis.
Demo UI: Simple interface for testing and synthesizing speech interactively.

Repository Structure

audio/                # Audio preprocessing utilities
config/               # Hyperparameters and configuration files
ecapa/                # ECAPA-TDNN speaker encoder
embeddings/           # Precomputed speaker embeddings
hifigan/              # HiFi-GAN vocoder for waveform synthesis
lexicon/              # Phoneme lexicon
model/                # FastSpeech2 model definitions
similarity/           # Voice similarity metrics and utilities
text/                 # Text processing utilities
transformer/          # Transformer models and support
unseenembeddings/     # Support for unseen speaker embeddings
utils/                # Shared utilities
train.py              # Train TTS and speaker models
generate.py           # Speech synthesis / inference
evaluate.py           # Evaluation scripts
ui.py                 # Demo UI for testing
requirements-*.txt    # System dependencies
README.md             # Documentation (this file)

Installation

Clone the repository:

git clone https://github.com/sandip-sharma1/Voice_cloning_system_using_Fastspeech2-ECAPA-TDNN.git
cd Voice_cloning_system_using_Fastspeech2-ECAPA-TDNN

Set up a Python virtual environment:

python -m venv venv
# Activate environment
source venv/bin/activate   # Linux / macOS
venv\Scripts\activate      # Windows

Install dependencies:

pip install -r requirements-linux.txt  # Linux
pip install -r requirements-windows.txt # Windows

Model Overview

FastSpeech2 (TTS)
Non-autoregressive architecture for fast, parallel speech generation.
Predicts mel-spectrograms from text input.
Requires aligned datasets for training (phoneme durations, etc.).


ECAPA-TDNN (Speaker Embeddings)
Extracts speaker identity features from reference audio.
Enables accurate voice cloning across multiple speakers.
High performance in speaker recognition tasks.


Usage
1. Prepare alignment
python ./prepare_align.py ./config/LibriTTS/preprocess.yaml

2. Preprocess dataset
python ./preprocess.py ./config/LibriTTS/preprocess.yaml

3. Train model
python ./train.py -p ./config/LibriTTS/preprocess.yaml \
                  -m ./config/LibriTTS/model.yaml \
                  -t ./config/LibriTTS/train.yaml

4. Inference / Generate speech
python generate.py --restore_step 500000 \
                   --mode single \
                   --text "Thank you" \
                   --speaker_emb embeddings/LibriTTS/1.pt \
                   -p config/LibriTTS/preprocess.yaml \
                   -m config/LibriTTS/model.yaml \
                   -t config/LibriTTS/train.yaml \
                   --pitch_control 1.0 \
                   --energy_control 1.0 \
                   --duration_control 1.0

5. Launch Demo UI
python ui.py

Supported Datasets:

LibriTTS – clean, multi-speaker TTS dataset.
Additional high-quality datasets can be adapted.
Ensure consistent sampling rate (e.g., 22,050 Hz) and aligned transcripts.

Tips & Notes
Precompute speaker embeddings before inference.
Ensure phoneme alignment during TTS training for best results.
Use HiFi-GAN vocoder for waveform synthesis.
Training on clean, high-quality data improves voice cloning accuracy.
Modify config/ files to match dataset paths and hyperparameters.


Acknowledgements
FastSpeech2 — Microsoft Research
ECAPA-TDNN — State-of-the-art speaker embedding model
Open-source community contributions for datasets and training utilities