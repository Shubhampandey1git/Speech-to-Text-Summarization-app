# How to Impement ASR Inference
# 1. Load model (Whisper)
# 2. Load dataset (CSV)
# 3. Loop over samples
# 4. Generate transcript
# 5. Let it run overnight on RTX 3036 it will take around 0.5 - 0.6 seconds per sample (depends on length) for 1000 samples it will take around 8-10 minutes
# 6. Save predictions

# =========================
# FAST + CORRECT ASR INFERENCE (FASTER-WHISPER)
# =========================

import csv
import os
import pandas as pd
from tqdm import tqdm
from faster_whisper import WhisperModel

# =========================
# CONFIG
# =========================
MODEL_SIZE = "base"
DEVICE = "cuda"
BASE_DIR = r"E:\Final Year Project\Speech-to-Text Summarization System for Smart Note-Taking"

SAVE_EVERY = 1000


# =========================
# LOAD MODEL
# =========================
def load_model():
    print("[INFO] Loading Faster-Whisper model...")
    model = WhisperModel(
        MODEL_SIZE,
        device=DEVICE,
        compute_type="int8_float16"
    )
    return model


# =========================
# RUN INFERENCE
# =========================
def run_inference(input_csv, output_csv, language=None):

    print(f"\n[INFO] Loading dataset: {input_csv}")

    df = pd.read_csv(input_csv)

    assert "audio" in df.columns and "text" in df.columns

    # -------------------------
    # FIX PATHS
    # -------------------------
    df["audio"] = df["audio"].apply(
        lambda x: str(x).replace("../", "").replace("\\", "/").strip()
    )

    df["audio"] = df["audio"].apply(
        lambda x: os.path.normpath(os.path.join(BASE_DIR, x))
    )

    # -------------------------
    # REMOVE MISSING FILES
    # -------------------------
    df = df[df["audio"].apply(os.path.exists)].reset_index(drop=True)

    print(f"[INFO] Valid samples: {len(df)}")

    df_full = df.copy()

    # -------------------------
    # RESUME SUPPORT
    # -------------------------
    predictions = []
    start_idx = 0

    if os.path.exists(output_csv):
        print("[INFO] Resuming from existing file...")

        df_existing = pd.read_csv(output_csv)

        if "prediction" in df_existing.columns:
            df_existing = df_existing.dropna(subset=["prediction"])

            predictions = df_existing["prediction"].astype(str).tolist()
            start_idx = len(predictions)

            df = df.iloc[start_idx:].reset_index(drop=True)

            print(f"[INFO] Resuming from sample {start_idx}")

    # -------------------------
    # LOAD MODEL
    # -------------------------
    model = load_model()

    print("[INFO] Starting inference...")

    paths = df["audio"].tolist()

    for i, path in enumerate(tqdm(paths)):

        try:
            segments, _ = model.transcribe(
                path,
                language=language,
                beam_size=5,
                task="transcribe",
                condition_on_previous_text=False,
                vad_filter=True,
                temperature=0.0
            )

            text = " ".join([seg.text for seg in segments]).strip()

        except Exception as e:
            print(f"[ERROR] {path}: {e}")
            text = ""

        predictions.append(text)

        # -------------------------
        # CHECKPOINT SAVE (FIXED)
        # -------------------------
        if len(predictions) % SAVE_EVERY == 0:

            current_len = len(predictions)

            df_partial = df_full.iloc[:current_len].copy()
            df_partial["prediction"] = predictions

            os.makedirs(os.path.dirname(output_csv), exist_ok=True)

            df_partial.to_csv(
                output_csv,
                index=False,
                encoding="utf-8",
                escapechar="\\",
                quoting=csv.QUOTE_ALL
            )

            print(f"[CHECKPOINT] Saved {current_len} samples")

    # -------------------------
    # FINAL SAVE (FIXED)
    # -------------------------
    final_len = len(predictions)

    df_final = df_full.iloc[:final_len].copy()
    df_final["prediction"] = predictions

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    df_final.to_csv(
        output_csv,
        index=False,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_ALL
    )

    print(f"[INFO] Completed and saved to: {output_csv}")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    print("\n===== HINDI INFERENCE START =====")

    run_inference(
        input_csv=r"data/processed/asr/hindi/train.csv",
        output_csv=r"experiments/asr/whisper_baseline/hindi_whisper_train.csv",
        language="hi"
    )

    print("\n===== HINDI DONE =====")

    print("\n===== ENGLISH INFERENCE START =====")

    run_inference(
        input_csv=r"data/processed/asr/english/train.csv",   # FIXED PATH
        output_csv=r"experiments/asr/whisper_baseline/english_whisper_train.csv",
        language="en"
    )

    print("\n===== ENGLISH DONE =====")