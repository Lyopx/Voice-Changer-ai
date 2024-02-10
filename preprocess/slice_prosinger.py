import librosa  # Optional. Use any library you like to read audio files.
import soundfile  # Optional. Use any library you like to write audio files.
import sys
import os
sys.path.append("../")
from utils.slicer2 import Slicer
from config import dataset2path
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--singer", type=str, default="李健")

args = parser.parse_args()
singer = args.singer
PROSINGER_PATH = dataset2path["ProSinger"]
SINGER_PATH = os.path.join(PROSINGER_PATH, "vocal", singer)
save_dir = os.path.join(PROSINGER_PATH, f"segments/{singer}")


if __name__ == "__main__":

    for song in os.listdir(SINGER_PATH):
        song_id = song.split(".")[0]

        audio, sr = librosa.load(f'{PROSINGER_PATH}/vocal/{singer}/{song}', sr=None, mono=False)  # Load an audio file with librosa.
        slicer = Slicer(
            sr=sr,
            threshold=-40,
            min_length=5000,
            min_interval=300,
            hop_size=10,
            max_sil_kept=500
        )
        chunks = slicer.slice(audio)
        
        for i, chunk in enumerate(chunks):
            if len(chunk.shape) > 1:
                chunk = chunk.T  # Swap axes if the audio is stereo.
            os.makedirs(save_dir, exist_ok=True)

            soundfile.write(f'{save_dir}/{song_id}_{i}.wav', chunk, sr)  # Save sliced audio files with soundfile.