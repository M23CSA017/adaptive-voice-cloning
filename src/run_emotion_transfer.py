import os
import argparse
from glob import glob
import torchaudio

import sys
sys.path.append("/home/m23csa017/adaptive-voice-cloning/openvoice/openvoice")

from api import TTSInference
import torch

TTS_INPUT_DIR = "/home/m23csa017/adaptive-voice-cloning/src/results/multi_speaker_tts"
CREMAD_REF_DIR = "/home/m23csa017/adaptive-voice-cloning/data/crema-d/AudioWAV"
OUTPUT_DIR = "/home/m23csa017/adaptive-voice-cloning/src/results/emotion_transferred"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_transfer(source_wav, reference_wav, output_path):
    api = TTSInference()
    audio_out = api.synthesize(
        source_wav=source_wav,
        reference_wav=reference_wav,
        custom_mode=True
    )
    torchaudio.save(output_path, torch.tensor(audio_out).unsqueeze(0), 24000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emotion", type=str, default="ANG", help="Emotion code (ANG, HAP, FEA, SAD, etc.)")
    parser.add_argument("--num_files", type=int, default=5, help="How many TTS files to process")
    args = parser.parse_args()

    tts_files = sorted(glob(os.path.join(TTS_INPUT_DIR, "*.wav")))[:args.num_files]
    emotion_clips = sorted(glob(os.path.join(CREMAD_REF_DIR, f"*_{args.emotion}_*.wav")))

    if not emotion_clips:
        print(f"[ERROR] No emotion reference clips found for: {args.emotion}")
        exit(1)

    reference_wav = emotion_clips[0]
    print(f"[INFO] Using reference emotion clip: {os.path.basename(reference_wav)}")

    for tts_path in tts_files:
        name = os.path.splitext(os.path.basename(tts_path))[0]
        output_path = os.path.join(OUTPUT_DIR, f"{name}_{args.emotion.lower()}.wav")
        print(f"[RUN] Transferring emotion to {name}...")
        run_transfer(source_wav=tts_path, reference_wav=reference_wav, output_path=output_path)

    print("✅ Emotion transfer completed.")
