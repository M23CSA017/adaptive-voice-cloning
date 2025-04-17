# Adaptive Voice Cloning with Emotion & Accent Transfer

This project presents an advanced end-to-end voice cloning system that not only mimics a speaker's voice using Tortoise-TTS, but also modulates emotion (e.g., happy, sad, angry) and accent (e.g., Nigerian, Scottish) using a fine-tuned speaker encoder and OpenVoice's ToneColorConverter.

## Features

- High-quality Voice Cloning using Tortoise-TTS  
- Emotion Control via CREMA-D reference clips  
- Accent Transfer using learned accent embeddings from CommonVoice  
- Gradio UI for easy interaction

## 1. Voice Cloning

Uses Tortoise-TTS to clone a speaker’s voice using a few reference clips.

```bash
python tortoise/do_tts.py --text "..." --voice p225 --preset fast
```

## 2. Emotion Control

Prepare 5 reference clips per emotion from the CREMA-D dataset.

Place them under the following directory structure:
tortoise/voices/<speaker_id>_<emotion>/ref1.wav ... ref5.wav


Then, run the following scripts:

```bash
python prepare_emotion_refs.py        # Creates folders for angry, happy, sad, etc.
python generate_all_emotions.py       # Synthesizes text with all emotions
```


## 3. Accent Transfer

Fine-tune a speaker encoder (Wav2Vec2 + projection head) on CommonVoice accent data.

Generate average speaker embeddings for each target accent.

Use OpenVoice's ToneColorConverter to transfer the accent onto the generated emotional speech.

```bash
python finetune_se_extractor.py       # Train the SE model
python run_accent_transfer_v1.py      # Create accent embeddings
python infer_accent_file.py           # Run inference on generated emotional audio
```

## Gradio Interface

Launch the UI using the following command:

```bash
python app.py
```

### Inputs

- Reference Audio (Your Voice)
- Text to be spoken
- Emotion (e.g., happy, sad, angry, fear, neutral)
- Accent (e.g., Canadian English, Nigerian Accent, Scottish English, etc.)

### Output

- Speech output with the selected emotion and accent applied


