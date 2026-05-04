import os
import torch
import torchaudio
import pandas as pd
from datasets import Dataset
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer
)

# -------------------------
# CONFIG
# -------------------------

BASE_DIR = r"E:/Final Year Project/Speech-to-Text Summarization System for Smart Note-Taking"

TRAIN_CSV = os.path.join(BASE_DIR, "data/processed/asr/english/train.csv")
DEV_CSV   = os.path.join(BASE_DIR, "data/processed/asr/english/test.csv")

MODEL_NAME = "facebook/wav2vec2-base-960h"
OUTPUT_DIR = os.path.join(BASE_DIR, "models/asr/english")

SAMPLING_RATE = 16000

# -------------------------
# FIX PATHS
# -------------------------

def fix_path(p):
    p = str(p).strip().replace("\\", "/")

    if p.startswith("../"):
        p = p[3:]

    if not os.path.isabs(p):
        p = os.path.join(BASE_DIR, p)

    return os.path.normpath(p)

# -------------------------
# AUDIO LOADING
# -------------------------

def load_audio(path):
    speech, sr = torchaudio.load(path)

    if speech.shape[0] > 1:
        speech = speech.mean(dim=0)

    if sr != SAMPLING_RATE:
        speech = torchaudio.functional.resample(speech, sr, SAMPLING_RATE)

    return speech.numpy()

# -------------------------
# DATA COLLATOR (LAZY LOAD)
# -------------------------

class DataCollatorCTC:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        input_values = []
        labels = []

        for f in features:
            speech = load_audio(f["audio"])

            inputs = self.processor(
                speech,
                sampling_rate=SAMPLING_RATE
            )

            with self.processor.as_target_processor():
                label = self.processor(f["text"]).input_ids

            input_values.append(inputs.input_values[0])
            labels.append(label)

        batch = self.processor.pad(
            {"input_values": input_values},
            padding=True,
            return_tensors="pt"
        )

        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                {"input_ids": labels},
                padding=True,
                return_tensors="pt"
            )

        batch["labels"] = labels_batch["input_ids"].masked_fill(
            labels_batch["attention_mask"].ne(1), -100
        )

        return batch

# -------------------------
# DURATION FILTER
# -------------------------

def get_duration(path):
    try:
        info = torchaudio.info(path)
        return info.num_frames / info.sample_rate
    except:
        return 0

# -------------------------
# TRAINING ARGS
# -------------------------

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,

    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,

    learning_rate=1e-5,
    warmup_steps=200,
    max_steps=1500,

    evaluation_strategy="steps",
    eval_steps=200,
    logging_steps=50,
    save_steps=500,

    fp16=True,
    remove_unused_columns=False,
    dataloader_num_workers=2,
    report_to="none",
    save_total_limit=2,
)

# -------------------------
# MAIN
# -------------------------

def main():
    df_train = pd.read_csv(TRAIN_CSV)
    df_dev   = pd.read_csv(DEV_CSV)

    # Fix paths
    df_train["audio"] = df_train["audio"].apply(fix_path)
    df_dev["audio"]   = df_dev["audio"].apply(fix_path)

    print("\n🔍 Checking paths...")
    for i in range(3):
        path = df_train["audio"].iloc[i]
        print(path, "->", os.path.exists(path))
    print("")

    # Duration filter
    print("⏳ Calculating durations...")
    df_train["duration"] = df_train["audio"].apply(get_duration)
    df_dev["duration"]   = df_dev["audio"].apply(get_duration)

    df_train = df_train[(df_train["duration"] >= 2) & (df_train["duration"] <= 8)]
    df_dev   = df_dev[(df_dev["duration"] >= 2) & (df_dev["duration"] <= 8)]

    df_train = df_train.reset_index(drop=True)
    df_dev   = df_dev.reset_index(drop=True)

    print("After filtering:", len(df_train), len(df_dev))

    # Convert (ONLY paths, no audio stored)
    train_dataset = Dataset.from_pandas(df_train)
    dev_dataset   = Dataset.from_pandas(df_dev)

    # Model
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2ForCTC.from_pretrained(MODEL_NAME)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    model.freeze_feature_encoder()

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        data_collator=DataCollatorCTC(processor),
        tokenizer=processor.feature_extractor,
    )

    # Train
    trainer.train()

    # Save
    final_path = os.path.join(OUTPUT_DIR, "final")
    trainer.save_model(final_path)
    processor.save_pretrained(final_path)

    print("✅ Model saved at:", final_path)


# -------------------------
# ENTRY POINT
# -------------------------

if __name__ == "__main__":
    main()