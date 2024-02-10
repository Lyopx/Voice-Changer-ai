import whisper
import torch
import os
import json
import numpy as np
import pickle
from tqdm import tqdm
import sys

sys.path.append("../")
from config import data_path, dataset2wavpath, WHISPER_SEQ, WHISPER_DIM


def whisper_encoder(audio_paths):
    batch = len(audio_paths)
    batch_mel = torch.zeros((batch, 80, 3000), dtype=torch.float, device=model.device)

    for i, audio_path in enumerate(audio_paths):
        # (48000,)
        audio = whisper.load_audio(str(audio_path))
        audio = whisper.pad_or_trim(audio)

        # (80, 3000)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        batch_mel[i] = mel

    with torch.no_grad():
        # (batch, 1500, 1024)
        features = model.embed_audio(batch_mel)

    return features.cpu().detach().numpy()


def get_mapped_whisper_features(dataset, dataset_type, raw_whisper_features):
    MCEP_dir = os.path.join(data_path, dataset, "MCEP")
    with open(os.path.join(MCEP_dir, "{}.pkl".format(dataset_type)), "rb") as f:
        mceps = pickle.load(f)
    print("MCEPs: {}, mceps[0] = {}".format(len(mceps), mceps[0].shape))

    whisper_features = []
    for index, mcep in enumerate(tqdm(mceps)):
        sz = len(mcep)

        # truncate if too long
        if sz > WHISPER_SEQ:
            mcep = mcep[:WHISPER_SEQ]
            sz = WHISPER_SEQ

        # (1500, 1024)
        raw_feats = raw_whisper_features[index]

        feats = np.zeros((sz, WHISPER_DIM), dtype=float)
        for i in range(sz):
            feats[i] = raw_feats[int(i / 2)]
        whisper_features.append(feats)

    return whisper_features


def extract_whisper_features(dataset, dataset_type, batch_size=80):
    print("-" * 20)
    print("Dataset: {}, {}".format(dataset, dataset_type))

    data_dir = os.path.join(data_path, dataset)
    os.makedirs(data_dir, exist_ok=True)

    wave_dir = dataset2wavpath[dataset]
    with open(os.path.join(data_dir, "{}.json".format(dataset_type)), "r") as f:
        datasets = json.load(f)

    # Extract raw features: (sz, 1500, 1024)
    print("\nExtracting raw whisper features...")
    whisper_features = np.zeros((len(datasets), WHISPER_SEQ, WHISPER_DIM), dtype=float)
    audio_paths = [
        os.path.join(wave_dir, "{}.wav".format(utt["Uid"])) for utt in datasets
    ]
    if dataset == "M4Singer" or dataset == "ProSinger":
        audio_paths = [os.path.join(wave_dir, utt["Path"]) for utt in datasets]

    start = 0
    end = 0
    while end <= len(audio_paths):
        start = end
        end = start + batch_size
        print("{}/{}...".format(min(len(audio_paths), end), len(audio_paths)))

        whisper_features[start:end] = whisper_encoder(audio_paths[start:end])

    # Mapping to mcep's lengths
    print("\nTransform to mapped features...")
    whisper_features = get_mapped_whisper_features(
        dataset, dataset_type, whisper_features
    )

    # Save
    output_dir = os.path.join(data_dir, "Whisper")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "{}.pkl".format(dataset_type)), "wb") as f:
        pickle.dump(whisper_features, f)


if __name__ == "__main__":
    print("Loading Model...")
    model = whisper.load_model("medium")
    if torch.cuda.is_available():
        print("Using GPU...\n")
        model = model.cuda()
    else:
        print("Using CPU...\n")

    model = model.eval()

    # extract_whisper_features("Opencpop", "test")
    # extract_whisper_features("Opencpop", "train")
    # extract_whisper_features("M4Singer", "test")
    extract_whisper_features("ProSinger", "train")
    extract_whisper_features("ProSinger", "test")
