import os
import shutil
from glob import glob
import random
import subprocess

CREMA_DIR = "/home/m23csa017/adaptive-voice-cloning/data/crema-d/AudioWAV"
TARGET_DIR = "/home/m23csa017/adaptive-voice-cloning/openvoice/tortoise-tts/tortoise/voices"

# Emotions to use (CREMA-D codes to folder suffix)
EMOTION_MAP = {
    "NEU": "neutral",
    "ANG": "angry",
    "HAP": "happy",
    "SAD": "sad"
}

# Target speakers
SPEAKERS = [f"p22{i}" for i in range(1, 25)] 

# Number of reference clips per emotion folder
N_CLIPS = 5

def convert_with_ffmpeg(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    subprocess.run([
        "ffmpeg", "-y", "-i", src,
        "-ar", "24000", "-ac", "1", dst
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

for speaker in SPEAKERS:
    for emo_code, emo_name in EMOTION_MAP.items():
        folder_name = f"{speaker}_{emo_name}"
        folder_path = os.path.join(TARGET_DIR, folder_name)
        os.makedirs(folder_path, exist_ok=True)

        clips = sorted(glob(os.path.join(CREMA_DIR, f"*_{emo_code}_*.wav")))
        selected = random.sample(clips, N_CLIPS)

        for idx, src in enumerate(selected):
            dst = os.path.join(folder_path, f"{emo_name}_{idx+1}.wav")
            convert_with_ffmpeg(src, dst)

print("✅ Emotion reference folders created with 24kHz mono audio.")
