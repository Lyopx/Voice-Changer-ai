import random
import os
import json
from collections import defaultdict
import sys

sys.path.append("../")
from config import data_path, dataset2path, NUMS_OF_SINGER


def m4singer_statistics():
    singers = []
    songs = []
    singer2songs = defaultdict(lambda: defaultdict(list))
    for utt in meta:
        p, s, uid = utt["item_name"].split("#")
        singers.append(p)
        songs.append(s)
        singer2songs[p][s].append(uid)

    unique_singers = list(set(singers))
    unique_songs = list(set(songs))
    unique_singers.sort()
    unique_songs.sort()

    print(
        "M4Singer: {} singers, {} songs ({} unique)".format(
            len(unique_singers), len(songs), len(unique_songs)
        )
    )
    print("Singers: {}".format(unique_singers))
    return singer2songs


if __name__ == "__main__":
    # Load
    m4singer_dir = dataset2path["M4Singer"]
    meta_file = os.path.join(m4singer_dir, "meta.json")
    with open(meta_file, "r") as f:
        meta = json.load(f)

    singer2songs = m4singer_statistics()

    # We select 5 utterances randomly for every singer
    test = []
    for singer, songs in singer2songs.items():
        song_names = list(songs.keys())

        for _ in range(NUMS_OF_SINGER):
            chosen_song = random.sample(song_names, 1)[0]
            chosen_uid = random.sample(songs[chosen_song], 1)[0]

            res = {"Singer": singer, "Song": chosen_song, "Uid": chosen_uid}
            res["Path"] = "{}#{}/{}.wav".format(singer, chosen_song, res["Uid"])
            assert os.path.exists(os.path.join(m4singer_dir, res["Path"]))
            test.append(res)

    print(
        "Every singer has {} utterances. The total size = {}".format(
            NUMS_OF_SINGER, len(test)
        )
    )

    # Save
    save_dir = os.path.join(data_path, "M4Singer")
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "test.json"), "w") as f:
        json.dump(test, f, indent=4, ensure_ascii=False)
