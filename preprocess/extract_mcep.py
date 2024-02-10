import diffsptk
import torch
import os
import pickle
from tqdm import tqdm
import sys

sys.path.append("../")
from config import data_path, MCEP_DIM, WORLD_SAMPLE_RATE
from extract_sp import extract_world_features_of_dataset


def SP2mgc(x, mcsize=MCEP_DIM - 1, fs=WORLD_SAMPLE_RATE):
    if fs == 44100:
        fft_size = 2048
        alpha = 0.77
    if fs == 16000:
        fft_size = 1024
        alpha = 0.58

    x = torch.as_tensor(x, dtype=torch.float)

    tmp = diffsptk.ScalarOperation("SquareRoot")(x)
    tmp = diffsptk.ScalarOperation("Multiplication", 32768.0)(tmp)
    mgc = diffsptk.MelCepstralAnalysis(
        cep_order=mcsize, fft_length=fft_size, alpha=alpha, n_iter=1
    )(tmp)
    return mgc.numpy()


def mgc2SP(x, mcsize=MCEP_DIM - 1, fs=WORLD_SAMPLE_RATE):
    if fs == 44100:
        fft_size = 2048
        alpha = 0.77
    if fs == 16000:
        fft_size = 1024
        alpha = 0.58

    x = torch.as_tensor(x, dtype=torch.float)

    tmp = diffsptk.MelGeneralizedCepstrumToSpectrum(
        alpha=alpha,
        cep_order=mcsize,
        fft_length=fft_size,
    )(x)
    tmp = diffsptk.ScalarOperation("Division", 32768.0)(tmp)
    sp = diffsptk.ScalarOperation("Power", 2)(tmp)
    return sp.double().numpy()


def extract_mcep_features(
    dataset, dataset_type, mcsize=MCEP_DIM - 1, fs=WORLD_SAMPLE_RATE
):
    print("-" * 20)
    print("Dataset: {}, {}".format(dataset, dataset_type))

    data_dir = os.path.join(data_path, dataset)
    os.makedirs(data_dir, exist_ok=True)

    # Extract SP features
    print("\nExtracting SP featuers...")
    sp_features = extract_world_features_of_dataset(dataset, dataset_type)

    # SP to MCEP
    print("\nTransform SP to MCEP...")
    mcep_features = [SP2mgc(sp, mcsize=mcsize, fs=fs) for sp in tqdm(sp_features)]

    # Save
    output_dir = os.path.join(data_dir, "MCEP")
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "{}.pkl".format(dataset_type)), "wb") as f:
        pickle.dump(mcep_features, f)


if __name__ == "__main__":
    # extract_mcep_features("Opencpop", "test")
    # extract_mcep_features("Opencpop", "train")
    # extract_mcep_features("M4Singer", "test")
    extract_mcep_features("ProSinger", "train")
    extract_mcep_features("ProSinger", "test")
