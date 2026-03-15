from tqdm import tqdm
import os
from text import _clean_text
import librosa
import numpy as np
from scipy.io import wavfile
import yaml

def prepare_align(config):
    in_dir = config["path"]["corpus_path"]
    out_dir = config["path"]["raw_path"]
    sampling_rate = config["preprocessing"]["audio"]["sampling_rate"]
    max_wav_value = config["preprocessing"]["audio"]["max_wav_value"]
    cleaners = config["preprocessing"]["text"]["text_cleaners"]

    for speaker in tqdm(os.listdir(in_dir)):

        for chapter in os.listdir(os.path.join(in_dir, speaker)):
            if "_embedding" in chapter:
                continue

            for file_name in os.listdir(os.path.join(in_dir, speaker, chapter)):
                if file_name[-4:] != ".wav":
                    continue

                base_name = file_name[:-4]
                text_path = os.path.join(in_dir, speaker, chapter, base_name + ".normalized.txt")
                wav_path = os.path.join(in_dir, speaker, chapter, base_name + ".wav")

                with open(text_path, "r", encoding="utf-8") as f:
                    text = f.readline().strip("\n")
                text = _clean_text(text, cleaners)

                os.makedirs(os.path.join(out_dir, speaker), exist_ok=True)
                wav, _ = librosa.load(wav_path, sr=sampling_rate)

                peak = np.max(np.abs(wav))
                if peak > 0:
                    wav = (wav / peak) * max_wav_value

                wavfile.write(
                    os.path.join(out_dir, speaker, base_name + ".wav"),
                    sampling_rate,
                    wav.astype(np.int16)
                )
                with open(
                    os.path.join(out_dir, speaker, base_name + ".txt"),
                    "w",
                    encoding="utf-8"
                ) as f1:
                    f1.write(text)

if __name__ == "__main__":
    config_path = "config/preprocess.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    prepare_align(config=config)
