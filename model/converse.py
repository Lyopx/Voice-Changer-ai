from tqdm import tqdm
import os
import numpy as np
import pickle
from argparse import ArgumentParser
import sys

sys.path.append("../")
sys.path.append("../preprocess")
from inference import get_uids, save_audio
from config import dataset2wavpath
import extract_sp
import extract_mcep


def converse_base_f0(target_singer_f0_file, ratio=0.25):
    # Loading target singer's F0 statistics
    with open(target_singer_f0_file, "rb") as f:
        mean, total_f0 = pickle.load(f)

    total_f0.sort()
    base = total_f0[int(len(total_f0) * ratio)]
    print("Target: mean = {}, ratio = {}, base = {}".format(mean, ratio, base))
    return base


def get_pred_audios(
    model_file,
    source_dataset,
    dataset_type,
    target_singer_f0_file,
    inference_dir,
):
    wave_dir = dataset2wavpath[source_dataset]
    paths = model_file.split("/")
    epoch = paths[-1].split(".")[0]

    output_dir = os.path.join(inference_dir, "{}_{}".format(dataset_type, epoch))
    if os.path.exists(output_dir):
        os.system("rm -r {}".format(output_dir))
    os.makedirs(output_dir)

    pred_file = os.path.join(inference_dir, "{}_{}.npy".format(dataset_type, epoch))
    # (dataset_sz, seq_len, MCEP_dim)
    pred = np.load(pred_file)
    print("File: {},\n Shape: {}\n".format(pred_file, pred.shape))

    # Loading base f0 (the lowest pitch in an utterance)
    ratio = 0.5
    target_base_f0 = converse_base_f0(target_singer_f0_file, ratio=ratio)

    # Predict
    uids = get_uids(source_dataset, dataset_type)
    if source_dataset == "M4Singer":
        uids, upaths = uids

    for i, uid in enumerate(tqdm(uids)):
        wave_file = os.path.join(wave_dir, "{}.wav".format(uid))
        if source_dataset == "M4Singer":
            wave_file = os.path.join(wave_dir, upaths[i])

        f0, sp, ap, fs = extract_sp.extract_world_features(wave_file)

        frame_len = min(len(f0), len(pred[i]))
        f0 = f0[:frame_len]
        sp = sp[:frame_len]
        ap = ap[:frame_len]

        # Get sp_pred
        mcep = pred[i][:frame_len]
        sp_pred = extract_mcep.mgc2SP(mcep)
        assert sp.shape == sp_pred.shape

        # Get transposed f0
        source_base_f0 = sorted([f for f in f0 if f != 0])
        source_base_f0 = source_base_f0[int(len(source_base_f0) * ratio)]
        f0_trans = f0 * (target_base_f0 / source_base_f0)
        print("File: {}, mapping = {}".format(uid, target_base_f0 / source_base_f0))

        # Synthesis by WORLD
        y_gt = extract_sp.world_synthesis(f0, sp, ap, fs)
        y_pred = extract_sp.world_synthesis(f0, sp_pred, ap, fs)
        y_pred_trans = extract_sp.world_synthesis(f0_trans, sp_pred, ap, fs)

        # save gt
        gt_file = os.path.join(output_dir, "{}.wav".format(uid))
        os.system("cp {} {}".format(wave_file, gt_file))
        # save WORLD synthesis gt
        world_gt_file = os.path.join(output_dir, "{}_world.wav".format(uid))
        save_audio(world_gt_file, y_gt, fs)
        # save WORLD synthesis pred
        world_pred_file = os.path.join(output_dir, "{}_pred.wav".format(uid))
        save_audio(world_pred_file, y_pred, fs)
        # save WORLD synthesis pred (f0 transposed)
        world_pred_trans_file = os.path.join(
            output_dir, "{}_pred_f0trans.wav".format(uid)
        )
        save_audio(world_pred_trans_file, y_pred_trans, fs)


if __name__ == "__main__":
    parser = ArgumentParser(description="Conversion for M4Singer")
    parser.add_argument("--source_dataset", type=str, default="M4Singer")
    parser.add_argument("--dataset_type", type=str, default="test")
    parser.add_argument(
        "--model_file", type=str, help="the checkpoint file of acoustic mapping model"
    )
    parser.add_argument("--target_singer_f0_file", type=str)
    parser.add_argument("--inference_dir", type=str)

    args = parser.parse_args()

    get_pred_audios(
        args.model_file,
        args.source_dataset,
        args.dataset_type,
        args.target_singer_f0_file,
        args.inference_dir,
    )
