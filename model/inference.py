import os
import json
import torchaudio
import torch
import random
import sys

sys.path.append("../")
from config import data_path, dataset2wavpath

sys.path.append("../preprocess")
import extract_sp
import extract_mcep


def get_uids(dataset, dataset_type):
    dataset_file = os.path.join(data_path, dataset, "{}.json".format(dataset_type))
    with open(dataset_file, "r") as f:
        utterances = json.load(f)

    if dataset == "M4Singer":
        uids = ["{}_{}_{}".format(u["Singer"], u["Song"], u["Uid"]) for u in utterances]
        upaths = [u["Path"] for u in utterances]
        return uids, upaths
    else:
        return [u["Uid"] for u in utterances]


def save_audio(path, waveform, fs):
    waveform = torch.as_tensor(waveform, dtype=torch.float32)
    if len(waveform.size()) == 1:
        waveform = waveform[None, :]
    # print("HERE: waveform", waveform.shape, waveform.dtype, waveform)
    torchaudio.save(path, waveform, fs, encoding="PCM_S", bits_per_sample=16)


def save_pred_audios_in_training(
    pred, args, loss, dataset_type, sample_sz=1, random_sampling=False
):
    dataset = args.dataset
    wave_dir = dataset2wavpath[dataset]

    if dataset == "ProSinger":
        wave_dir = os.path.join(wave_dir, args.singer, "segments") 

    output_dir = os.path.join(args.save, "{}".format(dataset_type))
    os.makedirs(output_dir, exist_ok=True)

    # Predict
    uids = get_uids(dataset, dataset_type)
    if random_sampling:
        sample_uids = random.sample(uids, sample_sz)
    else:
        sample_uids = uids[:sample_sz]

    for uid in sample_uids:
        i = uids.index(uid)
        wave_file = os.path.join(wave_dir, "{}.wav".format(uid))
        f0, sp, ap, fs = extract_sp.extract_world_features(wave_file)
        frame_len = len(f0)

        mcep = pred[i][:frame_len]
        sp_pred = extract_mcep.mgc2SP(mcep)
        assert sp.shape == sp_pred.shape

        y_gt = extract_sp.world_synthesis(f0, sp, ap, fs)
        y_pred = extract_sp.world_synthesis(f0, sp_pred, ap, fs)

        # save gt
        gt_file = os.path.join(output_dir, "{}_gt.wav".format(uid))
        os.system("cp {} {}".format(wave_file, gt_file))
        # save WORLD synthesis gt
        world_gt_file = os.path.join(output_dir, "{}_gt_world.wav".format(uid))
        save_audio(world_gt_file, y_gt, fs)
        # save WORLD synthesis pred
        world_pred_file = os.path.join(
            output_dir,
            "{}_pred_{}_loss{:.5f}.wav".format(uid, args.current_epoch, loss),
        )
        save_audio(world_pred_file, y_pred, fs)
