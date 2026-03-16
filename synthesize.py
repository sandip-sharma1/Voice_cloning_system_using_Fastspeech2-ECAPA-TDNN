import re
import argparse
from string import punctuation
import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from g2p_en import G2p
from pypinyin import pinyin, Style
from utils.model import get_model, get_vocoder
from utils.tools import to_device, synth_samples
from dataset import TextDataset
from text import text_to_sequence

# device = torch.device("mps" if torch.cuda.is_available() else "cpu")
device = "cpu" 

def read_lexicon(lex_path):
    lexicon = {}
    with open(lex_path) as f:
        for line in f:
            temp = re.split(r"\s+", line.strip("\n"))
            word = temp[0]
            phones = temp[1:]
            if word.lower() not in lexicon:
                lexicon[word.lower()] = phones
    return lexicon

def preprocess_english(text, preprocess_config):
    text = text.rstrip(punctuation)
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
    g2p = G2p()
    phones = []
    words = re.split(r"([,;.\-\?\!\s+])", text)
    for w in words:
        if w.lower() in lexicon:
            phones += lexicon[w.lower()]
        else:
            phones += list(filter(lambda p: p != " ", g2p(w)))
    phones = "{" + "}{".join(phones) + "}"
    phones = re.sub(r"\{[^\w\s]?\}", "{sp}", phones)
    phones = phones.replace("}{", " ")
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    return np.array(sequence)

def preprocess_mandarin(text, preprocess_config):
    lexicon = read_lexicon(preprocess_config["path"]["lexicon_path"])
    phones = []
    pinyins = [
        p[0]
        for p in pinyin(
            text, style=Style.TONE3, strict=False, neutral_tone_with_five=True
        )
    ]
    for p in pinyins:
        if p in lexicon:
            phones += lexicon[p]
        else:
            phones.append("sp")
    phones = "{" + " ".join(phones) + "}"
    print("Raw Text Sequence: {}".format(text))
    print("Phoneme Sequence: {}".format(phones))
    sequence = np.array(
        text_to_sequence(
            phones, preprocess_config["preprocessing"]["text"]["text_cleaners"]
        )
    )
    return np.array(sequence)

def load_speaker_embedding(embedding_path):
    if embedding_path.lower().endswith('.npy'):
        speaker_emb = np.load(embedding_path)
    elif embedding_path.lower().endswith('.pt'):
        speaker_emb = torch.load(embedding_path).cpu().numpy()
    else:
        raise ValueError("Unsupported embedding file format. Use .npy or .pt")
    
    if speaker_emb.ndim == 1:
        pass  # good [192]
    elif speaker_emb.ndim == 2:
        speaker_emb = speaker_emb.squeeze()  # [1,192] -> [192]
    else:
        raise ValueError(f"Unexpected embedding shape: {speaker_emb.shape}")
    
    # Return as [1, dim] tensor
    speakers = torch.from_numpy(speaker_emb).float().unsqueeze(0).to(device)
    print(f"Loaded speaker embedding shape: {speakers.shape}")
    return speakers

def synthesize(model, step, configs, vocoder, batchs, control_values, result_path=None, basenames=None):
    preprocess_config, model_config, train_config = configs
    pitch_control, energy_control, duration_control = control_values
    if result_path is None:
        result_path = train_config["path"]["result_path"]
    for batch in batchs:
        batch = to_device(batch, device, speaker_emb=preprocess_config["speaker_emb"])
        with torch.no_grad():
            # Forward
            output = model(
                *(batch[2:]),
                p_control=pitch_control,
                e_control=energy_control,
                d_control=duration_control
            )
            synth_samples(
                batch,
                output,
                vocoder,
                model_config,
                preprocess_config,
                result_path,
                basenames
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, required=True)
    parser.add_argument(
        "--mode",
        type=str,
        choices=["batch", "single"],
        required=True,
        help="Synthesize a whole dataset or a single sentence",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="path to a source file with format like train.txt and val.txt, for batch mode only",
    )
    parser.add_argument(
        "--text",
        type=str,
        default=None,
        help="raw text to synthesize, for single-sentence mode only",
    )
    parser.add_argument(
        "--speaker_emb",
        type=str,
        required=True,
        help="Path to the speaker embedding file (.npy or .pt) for voice cloning",
    )
    parser.add_argument(
        "-p",
        "--preprocess_config",
        type=str,
        required=True,
        help="path to preprocess.yaml",
    )
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml"
    )
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml"
    )
    parser.add_argument(
        "--pitch_control",
        type=float,
        default=1.0,
        help="control the pitch of the whole utterance, larger value for higher pitch",
    )
    parser.add_argument(
        "--energy_control",
        type=float,
        default=1.0,
        help="control the energy of the whole utterance, larger value for larger volume",
    )
    parser.add_argument(
        "--duration_control",
        type=float,
        default=1.0,
        help="control the speed of the whole utterance, larger value for slower speaking rate",
    )
    args = parser.parse_args()

    # Check source texts
    if args.mode == "batch":
        assert args.source is not None and args.text is None
    if args.mode == "single":
        assert args.source is None and args.text is not None

    # Read Config
    preprocess_config = yaml.load(
        open(args.preprocess_config, "r"), Loader=yaml.FullLoader
    )
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)

    # Get model
    model = get_model(args, configs, device, train=False)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Load speaker embedding
    speakers = load_speaker_embedding(args.speaker_emb)

    # Preprocess texts
    if args.mode == "batch":
        # Get dataset
        dataset = TextDataset(args.source, preprocess_config)
        batchs = DataLoader(
            dataset,
            batch_size=8,
            collate_fn=dataset.collate_fn,
        )
    if args.mode == "single":
        ids = raw_texts = [args.text[:100]]
        if preprocess_config["preprocessing"]["text"]["language"] == "en":
            texts = np.array([preprocess_english(args.text, preprocess_config)])
        elif preprocess_config["preprocessing"]["text"]["language"] == "zh":
            texts = np.array([preprocess_mandarin(args.text, preprocess_config)])
        text_lens = np.array([len(texts[0])])
        batchs = [(ids, raw_texts, speakers, texts, text_lens, max(text_lens))]

    control_values = args.pitch_control, args.energy_control, args.duration_control

    synthesize(model, args.restore_step, configs, vocoder, batchs, control_values)
    
