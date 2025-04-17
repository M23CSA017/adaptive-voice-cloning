# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# import argparse
# import torchaudio
# import torch
# import sys

# # Add Tortoise path
# sys.path.append("/home/m23csa017/adaptive-voice-cloning/openvoice/tortoise-tts")

# from tortoise.api import TextToSpeech, MODELS_DIR
# from tortoise.utils.audio import load_voices


# def generate_tts_for_all_speakers(output_dir, preset="fast", candidates=1):
#     tts = TextToSpeech(models_dir=MODELS_DIR, use_deepspeed=False, kv_cache=True, half=True)
#     os.makedirs(output_dir, exist_ok=True)

#     # Two short sentences to avoid autoregressive overflow
#     sentences = [
#         "This voice was generated as part of a speech understanding project.",
#         "It was conducted at the Indian Institute of Technology Jodhpur."
#     ]

#     voices_dir = "/home/m23csa017/adaptive-voice-cloning/openvoice/tortoise-tts/tortoise/voices"
#     base_speakers = [f"p22{i}" for i in range(1, 25)]
#     emotions = ["neutral", "angry", "happy", "sad"]
#     speakers = [f"{speaker}_{emo}" for speaker in base_speakers for emo in emotions]

#     for speaker_id in speakers:
#         print(f"[INFO] Generating voice for: {speaker_id}")
#         voice_samples, conditioning_latents = load_voices([speaker_id])

#         final_audio = []

#         for sentence in sentences:
#             gen = tts.tts_with_preset(
#                 sentence,
#                 k=candidates,
#                 voice_samples=voice_samples,
#                 conditioning_latents=conditioning_latents,
#                 preset=preset,
#                 return_deterministic_state=False
#             )

#             if isinstance(gen, list):
#                 gen = gen[0]
#             final_audio.append(gen.squeeze(0).cpu())

#         # Concatenate both sentence audio clips
#         # Concatenate both sentence audio clips
#         combined = torch.cat(final_audio, dim=-1)  # combined is 1D [samples]
#         combined = combined.unsqueeze(0)           # convert to [1, samples] for mono
#         assert combined.ndim == 2 and combined.shape[0] == 1, "Expected mono audio"

#         out_path = os.path.join(output_dir, f"{speaker_id}.wav")
#         torchaudio.save(out_path, combined, 24000)

#         # combined = torch.cat(final_audio, dim=-1)
#         # out_path = os.path.join(output_dir, f"{speaker_id}.wav")
#         # torchaudio.save(out_path, combined.unsqueeze(0), 24000)

#     print("✅ All voice clones generated.")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--output_dir', type=str, default='results/multi_speaker_tts', help='Where to save generated outputs.')
#     parser.add_argument('--preset', type=str, default='fast', help='TTS preset to use.')
#     parser.add_argument('--candidates', type=int, default=1, help='Number of output samples per speaker.')

#     args = parser.parse_args()

#     generate_tts_for_all_speakers(
#         output_dir=args.output_dir,
#         preset=args.preset,
#         candidates=args.candidates
#     )

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'  # Use GPU 1

import argparse
import torchaudio
import torch
import sys

# Add Tortoise path
sys.path.append("/home/m23csa017/adaptive-voice-cloning/openvoice/tortoise-tts")

from tortoise.api import TextToSpeech, MODELS_DIR
from tortoise.utils.audio import load_voices


def generate_tts_for_all_speakers(output_dir, preset="fast", candidates=1):
    tts = TextToSpeech(models_dir=MODELS_DIR, use_deepspeed=False, kv_cache=True, half=True)
    os.makedirs(output_dir, exist_ok=True)

    # Two short academic sentences
    sentences = [
        "It was conducted at the Indian Institute of Technology Jodhpur."
    ]

    # Reference voice folders
    voices_dir = "/home/m23csa017/adaptive-voice-cloning/openvoice/tortoise-tts/tortoise/voices"
    base_speakers = [f"p22{i}" for i in range(1, 25)]
    emotions = ["neutral", "angry", "happy", "sad"]
    speakers = [f"{speaker}_{emo}" for speaker in base_speakers for emo in emotions]

    for speaker_id in speakers:
        print(f"[INFO] Generating voice for: {speaker_id}")
        voice_samples, conditioning_latents = load_voices([speaker_id])
        final_audio = []

        for sentence in sentences:
            gen = tts.tts_with_preset(
                sentence,
                k=candidates,
                voice_samples=voice_samples,
                conditioning_latents=conditioning_latents,
                preset=preset,
                return_deterministic_state=False
            )

            if isinstance(gen, list):
                gen = gen[0]
            waveform = gen.squeeze()
            assert waveform.ndim == 1, f"Expected 1D waveform, got shape {waveform.shape}"
            final_audio.append(waveform.cpu())

        # Combine both into one [1, samples] tensor
        combined = torch.cat(final_audio, dim=-1).unsqueeze(0)
        assert combined.ndim == 2 and combined.shape[0] == 1, "Expected mono [1, samples]"

        out_path = os.path.join(output_dir, f"{speaker_id}.wav")
        torchaudio.save(out_path, combined, 24000)

    print("✅ All voice clones generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str, default='results/multi_speaker_tts', help='Where to save generated outputs.')
    parser.add_argument('--preset', type=str, default='fast', help='TTS preset to use.')
    parser.add_argument('--candidates', type=int, default=1, help='Number of output samples per speaker.')

    args = parser.parse_args()

    generate_tts_for_all_speakers(
        output_dir=args.output_dir,
        preset=args.preset,
        candidates=args.candidates
    )
