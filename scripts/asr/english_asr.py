import os
import torch
import torchaudio
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

# -------------------------
# CONFIG
# -------------------------

BASE_DIR = r"E:/Final Year Project/Speech-to-Text Summarization System for Smart Note-Taking"

MODEL_PATH = os.path.join(BASE_DIR, "models/asr/english/checkpoint-1000")
PROCESSOR_PATH = os.path.join(BASE_DIR, "models/asr/english/final")
CSV_PATH   = os.path.join(BASE_DIR, "data/processed/asr/english/test.csv")

OUTPUT_PATH = os.path.join(BASE_DIR, "experiments/asr/wav2vec2/english_results.csv")

SAMPLING_RATE = 16000

# -------------------------
# LOAD MODEL
# -------------------------

print("🚀 Loading model...")

processor = Wav2Vec2Processor.from_pretrained(PROCESSOR_PATH)
model = Wav2Vec2ForCTC.from_pretrained(MODEL_PATH)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# -------------------------
# FIX PATH
# -------------------------

def fix_path(p):
    p = str(p).strip().replace("\\", "/")

    if p.startswith("../"):
        p = p[3:]

    if not os.path.isabs(p):
        p = os.path.join(BASE_DIR, p)

    return os.path.normpath(p)

# -------------------------
# LOAD AUDIO
# -------------------------

def load_audio(path):
    speech, sr = torchaudio.load(path)

    if speech.shape[0] > 1:
        speech = speech.mean(dim=0)

    if sr != SAMPLING_RATE:
        speech = torchaudio.functional.resample(speech, sr, SAMPLING_RATE)

    return speech

# -------------------------
# LOAD DATA
# -------------------------

print("Loading dataset...")

df = pd.read_csv(CSV_PATH)
df["audio"] = df["audio"].apply(fix_path)

print("Total samples:", len(df))

# -------------------------
# INFERENCE LOOP
# -------------------------

predictions = []

for i, row in df.iterrows():
    speech = load_audio(row["audio"])

    input_values = processor(
        speech.squeeze(),
        sampling_rate=SAMPLING_RATE,
        return_tensors="pt"
    ).input_values.to(device)

    with torch.no_grad():
        logits = model(input_values).logits

    pred_ids = torch.argmax(logits, dim=-1)
    pred_text = processor.batch_decode(pred_ids)[0]

    predictions.append(pred_text)

    if i % 50 == 0:
        print(f"Processed {i}/{len(df)}")

        print("\n--- SAMPLE ---")
        print("GT:", row["text"])
        print("PR:", pred_text)

# -------------------------
# SAVE RESULTS
# -------------------------

df["prediction"] = predictions

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
df.to_csv(OUTPUT_PATH, index=False)

print("\n✅ Results saved at:", OUTPUT_PATH)