import pyworld as pw
import torchaudio
import os
import json
import numpy as np
from tqdm import tqdm
import pickle
import sys

sys.path.append("../")
from config import data_path, dataset2wavpath, WORLD_SAMPLE_RATE, WORLD_FRAME_SHIFT


def extract_world_features_of_dataset(
    dataset, dataset_type, frame_period=WORLD_FRAME_SHIFT
):
    data_dir = os.path.join(data_path, dataset)
    os.makedirs(data_dir, exist_ok=True)
    wave_dir = dataset2wavpath[dataset]

    # Dataset
    with open(os.path.join(data_dir, "{}.json".format(dataset_type)), "r") as f:
        datasets = json.load(f)

    # Save dir
    f0_dir = os.path.join(data_dir, "F0")
    os.makedirs(f0_dir, exist_ok=True)
    # sp_dir = os.path.join(data_dir, "SP")
    # os.makedirs(sp_dir, exist_ok=True)

    # Extract
    f0_features = []
    sp_features = []
    for utt in tqdm(datasets):
        uid = utt["Uid"]
        wave_file = os.path.join(wave_dir, "{}.wav".format(uid))

        if dataset == "M4Singer" or dataset == "ProSinger":
            wave_file = os.path.join(wave_dir, utt["Path"])

        f0, sp, _, _ = extract_world_features(wave_file, frame_period=frame_period)

        sp_features.append(sp)
        f0_features.append(f0)

    # # Save sp
    # with open(os.path.join(sp_dir, "{}.pkl".format(dataset_type)), "wb") as f:
    #     pickle.dump(sp_features, f)

    # F0 statistics
    f0_statistics_file = os.path.join(f0_dir, "{}_f0.pkl".format(dataset_type))
    f0_statistics(f0_features, f0_statistics_file)

    return sp_features


def extract_world_features(
    wave_file, fs=WORLD_SAMPLE_RATE, frame_period=WORLD_FRAME_SHIFT
):
    # waveform: (1, seq)
    waveform, sample_rate = torchaudio.load(wave_file)
    waveform = torchaudio.functional.resample(
        waveform, orig_freq=sample_rate, new_freq=fs
    )
    # x: (seq)
    x = np.array(waveform[0], dtype=np.double)

    _f0, t = pw.dio(x, fs, frame_period=frame_period)  # raw pitch extractor
    f0 = pw.stonemask(x, _f0, t, fs)  # pitch refinement
    sp = pw.cheaptrick(x, f0, t, fs)  # extract smoothed spectrogram
    ap = pw.d4c(x, f0, t, fs)  # extract aperiodicity

    return f0, sp, ap, fs


def world_synthesis(f0, sp, ap, fs, frame_period=WORLD_FRAME_SHIFT):
    y = pw.synthesize(
        f0, sp, ap, fs, frame_period=frame_period
    )  # synthesize an utterance using the parameters
    return y


def f0_statistics(f0_features, path):
    print("\nF0 statistics...")

    total_f0 = []
    for f0 in tqdm(f0_features):
        total_f0 += [f for f in f0 if f != 0]

    mean = sum(total_f0) / len(total_f0)
    print("Min = {}, Max = {}, Mean = {}".format(min(total_f0), max(total_f0), mean))

    with open(path, "wb") as f:
        pickle.dump([mean, total_f0], f)
