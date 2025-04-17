import os
import sys
import tempfile
import shutil
import subprocess
import gradio as gr
import torchaudio
import torch
import pickle
import glob
from pathlib import Path

# === Configuration ===
os.environ["CUDA_VISIBLE_DEVICES"] = "2"  # Set to your desired GPU
torch.set_default_dtype(torch.float32)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths (adjusted to match your environment)
TORTOISE_ROOT = "/home/m23csa017/adaptive-voice-cloning/openvoice/tortoise-tts/tortoise"
BASE_VOICE_DIR = "/home/m23csa017/adaptive-voice-cloning/openvoice/tortoise-tts/tortoise/voices"
OUTPUT_DIR = "/home/m23csa017/adaptive-voice-cloning/gradio_outputs"
SE_MODEL_PATH = "/home/m23csa017/adaptive-voice-cloning/openvoice/accent_transfer/se_extractor_finetuned.pth"
CONVERTER_CKPT_PATH = "/home/m23csa017/adaptive-voice-cloning/openvoice/checkpoints/checkpoints/checkpoints/converter"
ACCENT_EMBEDDING_DIR = "/home/m23csa017/adaptive-voice-cloning/openvoice/accent_transfer/embeddings"

# Temporary directory for Gradio uploads
tmp_dir = os.path.join(os.getcwd(), "gradio_tmp")
os.makedirs(tmp_dir, exist_ok=True)
tempfile.tempdir = tmp_dir

# Add OpenVoice to system path
sys.path.append(os.path.join(os.path.dirname(__file__), "openvoice"))
from openvoice.api import ToneColorConverter

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === SE Model Definition ===
class SimpleSE(torch.nn.Module):
    def __init__(self, embedding_dim=256):
        super().__init__()
        self.wav2vec = torchaudio.models.wav2vec2_base()
        for param in self.wav2vec.feature_extractor.parameters():
            param.requires_grad = False
        for param in self.wav2vec.encoder.transformer.layers[-4:]:
            param.requires_grad = True
        self.proj = torch.nn.Sequential(
            torch.nn.Linear(768, 512),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(512),
            torch.nn.Linear(512, embedding_dim)
        )

    def forward(self, x):
        features = self.wav2vec(x)[0]
        pooled = features.mean(dim=1)
        return self.proj(pooled)

# Load SE model
se_model = SimpleSE(embedding_dim=256).to(DEVICE)
try:
    se_model.load_state_dict(torch.load(SE_MODEL_PATH, map_location=DEVICE))
except RuntimeError as e:
    raise RuntimeError(f"Failed to load SE model: {e}")
se_model.eval()

# === Helper Functions ===
def setup_emotion_voice(audio_path, base_id, emotion):
    """Set up a voice folder for Tortoise-TTS with the given emotion."""
    folder = f"{base_id}_{emotion}" if emotion != "neutral" else base_id
    full_path = os.path.join(BASE_VOICE_DIR, folder)
    if os.path.exists(full_path):
        shutil.rmtree(full_path)
    os.makedirs(full_path, exist_ok=True)
    shutil.copy(audio_path, os.path.join(full_path, "ref1.wav"))
    return folder

def generate_emotional_audio(text, voice_id):
    """Generate audio with Tortoise-TTS for the specified voice and text."""
    cmd = [
        sys.executable, "-m", "tortoise.do_tts",
        "--text", text,
        "--voice", voice_id,
        "--preset", "fast",
        "--output_path", OUTPUT_DIR,
        "--candidates", "1"
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "/home/m23csa017/adaptive-voice-cloning/openvoice/tortoise-tts"

    try:
        result = subprocess.run(
            cmd,
            cwd="/home/m23csa017/adaptive-voice-cloning/openvoice/tortoise-tts",
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",  # Explicitly use UTF-8 to handle non-ASCII characters
            check=True
        )
        print("[STDOUT]", result.stdout)
        print("[STDERR]", result.stderr)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Tortoise-TTS failed: {e.stderr}")
    except UnicodeDecodeError as e:
        raise RuntimeError(f"Failed to decode Tortoise-TTS output: {e}")

    # Find the generated file
    pattern = os.path.join(OUTPUT_DIR, f"{voice_id}*.wav")
    matches = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not matches:
        raise FileNotFoundError(f"No generated WAV found matching: {pattern}")
    return matches[0]

def get_se_embedding(wav_path):
    """Extract speaker embedding from audio using the fine-tuned SE model."""
    try:
        waveform, sr = torchaudio.load(wav_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform[:1]
        waveform = waveform.squeeze(0)
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-7)
        with torch.no_grad():
            embedding = se_model(waveform.unsqueeze(0).to(DEVICE)).squeeze(0).cpu()
        return embedding
    except Exception as e:
        raise RuntimeError(f"Failed to extract SE embedding: {e}")

def apply_accent(audio_path, accent_name):
    """Apply accent transfer to the audio using ToneColorConverter."""
    try:
        tone_color_converter = ToneColorConverter(
            os.path.join(CONVERTER_CKPT_PATH, "config.json"),
            device=DEVICE
        )
        tone_color_converter.load_ckpt(os.path.join(CONVERTER_CKPT_PATH, "checkpoint.pth"))
    except Exception as e:
        raise RuntimeError(f"Failed to load ToneColorConverter: {e}")

    # Load target accent embedding
    tgt_embedding_path = os.path.join(ACCENT_EMBEDDING_DIR, f"{accent_name}_embedding.pkl")
    if not os.path.exists(tgt_embedding_path):
        raise FileNotFoundError(f"Accent embedding not found: {tgt_embedding_path}")
    
    try:
        with open(tgt_embedding_path, 'rb') as f:
            tgt_embedding = torch.tensor(pickle.load(f)).to(DEVICE)
    except Exception as e:
        raise RuntimeError(f"Failed to load accent embedding: {e}")

    # Get source embedding
    src_embedding = get_se_embedding(audio_path).to(DEVICE)

    # Save temporary source audio
    tmp_src_path = os.path.join(OUTPUT_DIR, f"tmp_source_{accent_name}.wav")
    try:
        waveform, sr = torchaudio.load(audio_path)
        if sr != 16000:
            waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        torchaudio.save(tmp_src_path, waveform, 16000)
    except Exception as e:
        raise RuntimeError(f"Failed to preprocess source audio: {e}")

    # Apply accent transfer
    output_path = os.path.join(OUTPUT_DIR, f"accented_{accent_name}.wav")
    try:
        tone_color_converter.convert(
            audio_src_path=tmp_src_path,
            src_se=src_embedding.unsqueeze(0).unsqueeze(-1),
            tgt_se=tgt_embedding.unsqueeze(0).unsqueeze(-1),
            output_path=output_path,
            tau=1.0,
            message=f"@Accent_{accent_name}"
        )
    except Exception as e:
        raise RuntimeError(f"Accent transfer failed: {e}")
    
    return output_path

def full_pipeline(text, audio, emotion, accent):
    """Main pipeline: Voice cloning with emotion injection and accent transfer."""
    if not text:
        raise ValueError("Text input is empty.")
    if audio is None:
        raise ValueError("Audio input not received. Please upload a valid file.")
    
    print(f"[INFO] Inputs: Text={text}, Audio={audio}, Emotion={emotion}, Accent={accent}")
    
    # Step 1: Set up emotion-specific voice folder
    emotion_voice_folder = setup_emotion_voice(audio, base_id="custom", emotion=emotion)
    print(f"[INFO] Emotion-specific voice folder: {emotion_voice_folder}")
    
    # Step 2: Generate emotional audio with Tortoise-TTS
    emotion_audio = generate_emotional_audio(text, emotion_voice_folder)
    print(f"[INFO] Emotion audio generated: {emotion_audio}")
    
    # Step 3: Apply accent transfer
    final_audio = apply_accent(emotion_audio, accent)
    print(f"[INFO] Accent applied: {final_audio}")
    
    return final_audio

# === Gradio Interface ===
if __name__ == "__main__":
    def gradio_wrapper(text, audio, emotion, accent):
        try:
            result = full_pipeline(text, audio, emotion, accent)
            return result, None
        except Exception as e:
            return None, f"Error: {str(e)}"

    interface = gr.Interface(
        fn=gradio_wrapper,
        inputs=[
            gr.Textbox(label="Input Text", placeholder="Enter the text to synthesize..."),
            gr.Audio(source="upload", type="filepath", label="Reference Audio"),
            gr.Dropdown(
                choices=["neutral", "happy", "angry", "sad", "fear"],
                label="Emotion",
                value="neutral"
            ),
            gr.Dropdown(
                choices=[
                    "canadian_english",
                    "filipino",
                    "nigerian_accent",
                    "scottish_english",
                    "united_states_english"
                ],
                label="Accent",
                value="united_states_english"
            )
        ],
        outputs=[
            gr.Audio(type="filepath", label="Generated Speech"),
            gr.Textbox(label="Error Message", visible=True)
        ],
        title="Adaptive Voice Cloning with Emotion and Accent",
        description=(
            "Upload a reference audio file, enter text, select an emotion, and choose an accent. "
            "The system will clone the voice, inject the selected emotion, and apply the chosen accent."
        ),
        allow_flagging="never"
    )
    interface.launch(server_name="0.0.0.0", server_port=7860)