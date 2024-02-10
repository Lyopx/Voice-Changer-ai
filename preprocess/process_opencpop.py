import json
from tqdm import tqdm
import os
import torchaudio
import sys

sys.path.append("../")
from config import data_path, dataset2path


def get_lines(file):
    with open(file, "r") as f:
        lines = f.readlines()
        lines = [l.strip() for l in lines]
    return lines


def get_uid2utt(file):
    lines = get_lines(file)

    uid2utt = []
    for l in tqdm(lines):
        items = l.split("|")
        uid = items[0]
        res = {"Uid": uid, "Text": items[1]}

        phonemes = items[2].split(" ")
        notes = items[3].split(" ")
        notes_durations = list(map(float, items[4].split(" ")))
        phonemes_durations = list(map(float, items[5].split(" ")))
        slur_notes = list(map(int, items[6].split(" ")))

        res["Phonemes"] = []
        for i, pho in enumerate(phonemes):
            r = dict()
            r["phoneme"] = pho
            r["duration"] = phonemes_durations[i]
            r["note"] = [notes[i], notes_durations[i], slur_notes[i]]
            res["Phonemes"].append(r)

        # Duration in wav files
        audio_file = os.path.join(opencpop_path, "segments/wavs/{}.wav".format(uid))
        waveform, sample_rate = torchaudio.load(audio_file)
        duration = waveform.size(-1) / sample_rate
        res["Duration"] = duration

        uid2utt.append(res)

    return uid2utt


if __name__ == "__main__":
    # Load
    opencpop_path = dataset2path["Opencpop"]
    train_file = os.path.join(opencpop_path, "segments", "train.txt")
    test_file = os.path.join(opencpop_path, "segments", "test.txt")

    # Process
    train = get_uid2utt(train_file)
    test = get_uid2utt(test_file)

    # Save
    save_dir = os.path.join(data_path, "Opencpop")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "train.json"), "w") as f:
        json.dump(train, f, indent=4, ensure_ascii=False)
    with open(os.path.join(save_dir, "test.json"), "w") as f:
        json.dump(test, f, indent=4, ensure_ascii=False)
