import os
import torch
import torchaudio
import pandas as pd
from transformers import AutoProcessor, AutoModelForCTC, Wav2Vec2Processor

# -------------------------
# CONFIG
# -------------------------

MODEL_PATH = "models/asr/hindi/final"   # your saved model
DEV_CSV    = "data/processed/asr/hindi/dev.csv"
BASE_DIR   = "E:/Final Year Project/Speech-to-Text Summarization System for Smart Note-Taking"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUTPUT_FILE = "experiments/wav2vec2/hindi_predictions.csv"

# -------------------------
# UTIL
# -------------------------

def fix_path(p):
    p = str(p).strip().replace("\\", "/")
    if p.startswith("../"):
        p = p[3:]
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(BASE_DIR, p))

def load_audio(path):
    speech, sr = torchaudio.load(path)

    if speech.shape[0] > 1:
        speech = speech.mean(dim=0, keepdim=True)

    if sr != 16000:
        speech = torchaudio.functional.resample(speech, sr, 16000)

    return speech.squeeze().numpy()

# -------------------------
# LOAD MODEL
# -------------------------

print("🚀 Loading model...")

processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
model = AutoModelForCTC.from_pretrained(MODEL_PATH).to(DEVICE)

model.eval()

# -------------------------
# LOAD DATA
# -------------------------

print("📂 Loading dev dataset...")

df = pd.read_csv(os.path.join(BASE_DIR, DEV_CSV))
df["audio"] = df["audio"].apply(fix_path)

print(f"Total samples: {len(df)}")

# -------------------------
# INFERENCE
# -------------------------

predictions = []

for i, row in df.iterrows():
    audio_path = row["audio"]

    try:
        speech = load_audio(audio_path)

        inputs = processor(
            speech,
            sampling_rate=16000,
            return_tensors="pt"
        ).input_values.to(DEVICE)

        with torch.no_grad():
            logits = model(inputs).logits

        pred_ids = torch.argmax(logits, dim=-1)

        pred_text = processor.batch_decode(pred_ids)[0]

    except Exception as e:
        print(f"❌ Error at {audio_path}: {e}")
        pred_text = ""

    predictions.append(pred_text)

    # progress
    if i % 50 == 0:
        print(f"Processed {i}/{len(df)}")
    
    if i < 5:
        print("\n--- SAMPLE ---")
        print("GT:", row["text"])
        print("PR:", pred_text)

# -------------------------
# SAVE RESULTS
# -------------------------

df["prediction"] = predictions

output_path = os.path.join(BASE_DIR, OUTPUT_FILE)
os.makedirs(os.path.dirname(output_path), exist_ok=True)
df.to_csv(output_path, index=False)

print("✅ Done!")
print("Saved at:", output_path)