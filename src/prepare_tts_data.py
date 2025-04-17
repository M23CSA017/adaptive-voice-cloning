# import os
# import subprocess
# from glob import glob
# from tqdm import tqdm

# # === CONFIG ===
# VCTK_FLAC_ROOT = "/home/m23csa017/adaptive-voice-cloning/data/vctk/wav48_silence_trimmed"
# OUTPUT_VOICES_DIR = "/home/m23csa017/adaptive-voice-cloning/data/voices"
# NUM_SPEAKERS = 10
# FILES_PER_SPEAKER = 5

# # === Ensure voices/ exists ===
# os.makedirs(OUTPUT_VOICES_DIR, exist_ok=True)

# # === Get list of speaker folders ===
# all_speakers = sorted([d for d in os.listdir(VCTK_FLAC_ROOT) if d.startswith('p')])[:NUM_SPEAKERS]

# print(f"Preparing {NUM_SPEAKERS} speakers with {FILES_PER_SPEAKER} files each...")

# for speaker_id in tqdm(all_speakers):
#     speaker_src_dir = os.path.join(VCTK_FLAC_ROOT, speaker_id)
#     speaker_out_dir = os.path.join(OUTPUT_VOICES_DIR, speaker_id)
#     os.makedirs(speaker_out_dir, exist_ok=True)

#     # Get .flac files (either mic1 or mic2)
#     flac_files = sorted(glob(os.path.join(speaker_src_dir, "*.flac")))[:FILES_PER_SPEAKER]

#     for i, flac_path in enumerate(flac_files):
#         out_wav_path = os.path.join(speaker_out_dir, f"{speaker_id}_{i+1}.wav")
#         cmd = [
#             "ffmpeg",
#             "-y",  # Overwrite
#             "-i", flac_path,
#             "-ar", "24000",  # Resample to 24kHz
#             out_wav_path
#         ]
#         subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

# print("✅ Voice folders prepared in:", os.path.abspath(OUTPUT_VOICES_DIR))

import os
import subprocess
from glob import glob
from tqdm import tqdm

# === CONFIG ===
VCTK_FLAC_ROOT = "/home/m23csa017/adaptive-voice-cloning/data/vctk/wav48_silence_trimmed"
OUTPUT_VOICES_DIR = "/home/m23csa017/adaptive-voice-cloning/data/voices"
NUM_SPEAKERS = 20
FILES_PER_SPEAKER = 10
MIC_SUFFIX = "mic1"  # Use mic1 files only for consistency

# === Ensure voices/ exists ===
os.makedirs(OUTPUT_VOICES_DIR, exist_ok=True)

# === Get list of speaker folders ===
all_speakers = sorted([d for d in os.listdir(VCTK_FLAC_ROOT) if d.startswith('p')])[:NUM_SPEAKERS]

print(f"Preparing {NUM_SPEAKERS} speakers with {FILES_PER_SPEAKER} mic1 files each...")

for speaker_id in tqdm(all_speakers):
    speaker_src_dir = os.path.join(VCTK_FLAC_ROOT, speaker_id)
    speaker_out_dir = os.path.join(OUTPUT_VOICES_DIR, speaker_id)
    os.makedirs(speaker_out_dir, exist_ok=True)

    # Pick first N mic1 files
    flac_files = sorted(glob(os.path.join(speaker_src_dir, f"*_{MIC_SUFFIX}.flac")))[:FILES_PER_SPEAKER]

    for i, flac_path in enumerate(flac_files):
        out_wav_path = os.path.join(speaker_out_dir, f"{speaker_id}_{i+1}.wav")
        cmd = [
            "ffmpeg",
            "-y",
            "-i", flac_path,
            "-ar", "24000",
            out_wav_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("✅ All speaker folders prepared in:", OUTPUT_VOICES_DIR)
