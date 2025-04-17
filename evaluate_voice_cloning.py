import os
import json
import torch
import torchaudio
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from python_speech_features import mfcc
from speechbrain.inference import EncoderClassifier
from app import full_pipeline, SimpleSE, SE_MODEL_PATH


# Import the pipeline function and module from your app.py
import app
from app import full_pipeline

# === Configuration Overrides ===
SOURCE_AUDIO   = "/home/m23csa017/adaptive-voice-cloning/openvoice/tortoise-tts/tortoise-tts/results/1001_0.wav"
INPUT_TEXT     = "This place is too far"
GENERATION_DIR = "/home/m23csa017/adaptive-voice-cloning/evaluation_outputs"
EVALUATION_DIR = GENERATION_DIR
EMOTIONS       = ["neutral", "happy", "angry", "sad", "fear"]
ACCENTS        = ["canadian_english", "filipino", "nigerian_accent", "scottish_english", "united_states_english"]

# Output filenames
OUTPUT_JSON      = os.path.join(EVALUATION_DIR, "evaluation_results.json")
SIMILARITY_TABLE = os.path.join(EVALUATION_DIR, "similarity_table.txt")
MCD_TABLE        = os.path.join(EVALUATION_DIR, "mcd_table.txt")

# Ensure directories exist
os.makedirs(GENERATION_DIR, exist_ok=True)
os.makedirs(EVALUATION_DIR, exist_ok=True)

# Monkey-patch app's OUTPUT_DIR so that full_pipeline writes into GENERATION_DIR
app.OUTPUT_DIR = GENERATION_DIR

# Device for torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Helper: extract speaker embedding via ECAPA-TDNN model
def extract_embedding(audio_path, spk_model):
    waveform, sr = torchaudio.load(audio_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    if waveform.shape[0] > 1:
        waveform = waveform[:1]  # use mono
    waveform = waveform.squeeze(0)
    waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-7)
    with torch.no_grad():
        embedding = spk_model(waveform.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
    return embedding.numpy()


# Helper: compute Mel-Cepstral Distortion
def compute_mcd(ref_path, gen_path):
    # Load and resample
    ref, sr = torchaudio.load(ref_path)
    gen, _ = torchaudio.load(gen_path)
    
    if sr != 16000:
        ref = torchaudio.transforms.Resample(sr, 16000)(ref)
        gen = torchaudio.transforms.Resample(sr, 16000)(gen)

    # Convert to mono and normalize together
    ref = ref[:1].mean(dim=0)
    gen = gen[:1].mean(dim=0)
    
    # Scale both signals to same RMS energy
    ref = ref / (ref.norm() + 1e-8)
    gen = gen / (gen.norm() + 1e-8)

    # Compute MFCCs with consistent parameters
    ref_mfcc = mfcc(ref.numpy(), 
                   samplerate=16000,
                   numcep=13,
                   nfilt=26,
                   nfft=512,
                   appendEnergy=True)
    
    gen_mfcc = mfcc(gen.numpy(),
                   samplerate=16000,
                   numcep=13,
                   nfilt=26,
                   nfft=512,
                   appendEnergy=True)

    # Align MFCC sequences (simple truncation for now)
    min_len = min(ref_mfcc.shape[0], gen_mfcc.shape[0])
    ref_mfcc = ref_mfcc[:min_len]
    gen_mfcc = gen_mfcc[:min_len]

    # Compute squared differences
    diff = ref_mfcc - gen_mfcc
    diff_sq = np.sum(diff ** 2, axis=1)  # Sum over MFCC coefficients

    # Compute MCD (correct scaling)
    mcd = (10 / np.log(10)) * np.mean(np.sqrt(2 * diff_sq))
    
    return float(mcd)


# Main evaluation function
def evaluate_all():
    # Initialize speaker embedding model once
    spk_model = SimpleSE(embedding_dim=256).to(DEVICE)
    spk_model.load_state_dict(torch.load(SE_MODEL_PATH, map_location=DEVICE))
    spk_model.eval()
    results = {"similarity": {}, "mcd": {}}

    # Iterate through all combinations
    for emotion in EMOTIONS:
        for accent in ACCENTS:
            key = f"{accent}_{emotion}"  # store key as accent_emotion
            print(f"[INFO] Generating + evaluating: {key}")

            # Run the pipeline: this writes files to GENERATION_DIR
            gen_wav = full_pipeline(
                text=INPUT_TEXT,
                audio=SOURCE_AUDIO,
                emotion=emotion,
                accent=accent
            )

            # Rename the generated file to accent_emotion.wav
            new_name = os.path.join(GENERATION_DIR, f"{accent}_{emotion}.wav")
            try:
                os.replace(gen_wav, new_name)
                gen_wav = new_name
            except Exception:
                # if rename fails, continue with original path
                pass

            # Compute cosine similarity
            ref_emb = extract_embedding(SOURCE_AUDIO, spk_model)
            gen_emb = extract_embedding(gen_wav, spk_model)
            sim = cosine_similarity(ref_emb.reshape(1, -1), gen_emb.reshape(1, -1))[0, 0]

            # Compute MCD
            mcd = compute_mcd(SOURCE_AUDIO, gen_wav)

            results["similarity"][key] = float(sim)
            results["mcd"][key]        = float(mcd)

    # Save JSON results
    with open(OUTPUT_JSON, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"[INFO] Saved results to {OUTPUT_JSON}")

    # Write markdown tables
    def write_table(metric, filename):
        header = ['Accent'] + EMOTIONS
        with open(filename, 'w') as f:
            f.write(f"# {metric.upper()}\n")
            f.write("| " + " | ".join(header) + " |\n")
            f.write("| " + " | ".join(['---'] * len(header)) + " |\n")
            for accent in ACCENTS:
                row = [accent]
                for emotion in EMOTIONS:
                    key = f"{accent}_{emotion}"
                    row.append(f"{results[metric].get(key, 0):.4f}")
                f.write("| " + " | ".join(row) + " |\n")
        print(f"[INFO] Wrote {metric} table to {filename}")

    write_table('similarity', SIMILARITY_TABLE)
    write_table('mcd',        MCD_TABLE)

if __name__ == '__main__':
    evaluate_all()
