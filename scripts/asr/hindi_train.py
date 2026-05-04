import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import torchaudio
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List
from datasets import Dataset, Audio
from transformers import (
    AutoProcessor,
    AutoModelForCTC,
    TrainingArguments,
    Trainer,
    TrainerCallback
)
import unicodedata
import re
import numpy as np

# -------------------------
# CONFIG
# -------------------------

MODEL_NAME   = "ai4bharat/indicwav2vec-hindi"
BASE_DIR     = "E:/Final Year Project/Speech-to-Text Summarization System for Smart Note-Taking"
TRAIN_CSV    = "data/processed/asr/hindi/clean/train.csv"
OUTPUT_DIR   = "models/asr/hindi"
CACHE_DIR    = "data/processed/asr/hindi/hf_cache"   # ← Processed features saved here
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"

MAX_DURATION = 8.0    # seconds - longer = more VRAM
DEBUG_SAMPLES = None   # set to None for full dataset

# -------------------------
# UTILITIES
# -------------------------

def fix_path(p):
    p = str(p).strip().replace("\\", "/")
    if p.startswith("../"):
        p = p[3:]
    if os.path.isabs(p):
        return os.path.normpath(p)
    return os.path.normpath(os.path.join(BASE_DIR, p))

def normalize_text(text):
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\u0900-\u097F\s]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def get_duration(path):
    try:
        info = torchaudio.info(path)
        return info.num_frames / info.sample_rate
    except:
        return 999.0

# -------------------------
# DATA COLLATOR
# -------------------------

@dataclass
class DataCollatorCTC:
    processor: AutoProcessor
    pad_id: int

    def __call__(self, features: List[Dict]):
        # Features already have input_values and labels from .map()
        input_values    = [torch.tensor(f["input_values"])    for f in features]
        attention_masks = [torch.tensor(f["attention_mask"])  for f in features]
        labels          = [torch.tensor(f["labels"])          for f in features]

        input_values_padded = torch.nn.utils.rnn.pad_sequence(
            input_values, batch_first=True, padding_value=0.0
        )
        attention_masks_padded = torch.nn.utils.rnn.pad_sequence(
            attention_masks, batch_first=True, padding_value=0
        )

        max_label_len = max(len(l) for l in labels)
        labels_padded = torch.full(
            (len(labels), max_label_len), -100, dtype=torch.long
        )
        for i, label in enumerate(labels):
            labels_padded[i, :len(label)] = label

        return {
            "input_values":  input_values_padded,
            "attention_mask": attention_masks_padded,
            "labels":        labels_padded,
        }

# -------------------------
# INFERENCE CALLBACK
# -------------------------

class InferenceCallback(TrainerCallback):
    def __init__(self, processor, eval_dataset, device):
        self.processor   = processor
        self.eval_dataset = eval_dataset
        self.device      = device

    def on_evaluate(self, args, state, control, **kwargs):
        model = kwargs["model"]
        model.eval()

        print(f"\n🔍 Inference @ step {state.global_step}:")
        print("-" * 60)

        for i in range(min(3, len(self.eval_dataset))):
            sample = self.eval_dataset[i]

            input_values = torch.tensor(
                sample["input_values"]
            ).unsqueeze(0).to(self.device)

            with torch.no_grad():
                logits = model(input_values).logits

            probs       = torch.softmax(logits, dim=-1)
            blank_id    = self.processor.tokenizer.pad_token_id
            blank_ratio = probs[0, :, blank_id].mean()

            pred_ids     = torch.argmax(logits, dim=-1)
            pred_ids_raw = pred_ids[0][:30].tolist()
            pred_text    = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)[0]
            pred_tok_ids = self.processor.tokenizer.encode(pred_text)
            gt_tok_ids   = sample["labels"][:20]

            print(f"\n  Sample {i+1}:")
            print(f"  GT text : {sample['text'][:60]}")
            print(f"  PR text : {pred_text[:60] if pred_text else '(empty)'}")
            print(f"  GT ids  : {gt_tok_ids[:20]}")
            print(f"  PR ids  : {pred_tok_ids[:20]}")
            print(f"  Raw IDs : {pred_ids_raw}")
            print(f"  Blank % : {blank_ratio:.1%}")

            unique_ids    = list(set(pred_ids_raw))
            unique_tokens = [
                self.processor.tokenizer.convert_ids_to_tokens(uid)
                for uid in unique_ids
            ]
            print(f"  Unique predicted tokens: {list(zip(unique_ids, unique_tokens))}")

        print("-" * 60)
        model.train()

# -------------------------
# MAIN
# -------------------------

if __name__ == "__main__":

    print("="*60)
    print("🚀 HINDI ASR - ai4bharat indicwav2vec-hindi")
    print("="*60)

    # -------------------------
    # LOAD & FILTER DATA
    # -------------------------

    print("\n[1/6] Loading data...")

    df = pd.read_csv(os.path.join(BASE_DIR, TRAIN_CSV))
    df["audio"] = df["audio"].apply(fix_path)
    df = df[df["audio"].apply(os.path.exists)].reset_index(drop=True)
    df["text"]  = df["text"].apply(normalize_text)
    df = df[df["text"].str.len() > 0].reset_index(drop=True)
    print(f"   Total valid samples: {len(df)}")

    # Filter by duration - avoids long audio blowing up VRAM
    print(f"   Filtering audio > {MAX_DURATION}s (checking durations)...")
    df["duration"] = df["audio"].apply(get_duration)
    df = df[df["duration"] <= MAX_DURATION].reset_index(drop=True)
    df = df.drop(columns=["duration"])
    print(f"   After duration filter: {len(df)}")

    # Debug subset
    if DEBUG_SAMPLES is not None:
        df = df.head(DEBUG_SAMPLES).copy()
        print(f"   Using debug subset: {len(df)} samples")

    # -------------------------
    # LOAD PROCESSOR
    # -------------------------

    print("\n[2/6] Loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_NAME)
    print(f"   Vocab size:   {len(processor.tokenizer)}")
    print(f"   Pad token ID: {processor.tokenizer.pad_token_id}")

    # -------------------------
    # BUILD HF DATASET & MAP
    # Processes audio once → saves to disk cache → streams during training
    # RAM usage stays low because HF datasets uses memory mapping
    # -------------------------

    print("\n[3/6] Building dataset with disk cache...")

    os.makedirs(os.path.join(BASE_DIR, CACHE_DIR), exist_ok=True)
    cache_path = os.path.join(BASE_DIR, CACHE_DIR)

    hf_dataset = Dataset.from_pandas(df[["audio", "text"]])
    hf_dataset = hf_dataset.shuffle(seed=42)
    hf_dataset = hf_dataset.train_test_split(test_size=0.1, seed=42)

    print(f"   Raw split → Train: {len(hf_dataset['train'])}, Eval: {len(hf_dataset['test'])}")

    def preprocess_fn(batch, processor):
        """
        Called once per sample during .map(), result saved to disk.
        During training, HF loads only the current batch from disk.
        RAM stays low regardless of dataset size.
        """
        audio_path = batch["audio"]

        speech, sr = torchaudio.load(audio_path)

        if speech.shape[0] > 1:
            speech = speech.mean(dim=0, keepdim=True)
        if sr != 16000:
            speech = torchaudio.functional.resample(speech, sr, 16000)

        speech = speech.squeeze().numpy()

        inputs = processor(
            speech,
            sampling_rate=16000,
            return_attention_mask=True,
        )

        label_ids = processor.tokenizer(batch["text"]).input_ids

        return {
            "input_values":   inputs.input_values[0],    # float32 array → saved to disk
            "attention_mask": inputs.attention_mask[0],  # int array → saved to disk
            "labels":         label_ids,                 # int list → saved to disk
            "text":           batch["text"],             # keep for inference display
        }

    print("   Running map() — this may take a few minutes (only happens once!)...")
    print("   Next run will load from cache instantly ✅")

    hf_dataset = hf_dataset.map(
        preprocess_fn,
        fn_kwargs={"processor": processor},
        remove_columns=["audio"],   # Don't need raw path after processing
        cache_file_names={
            "train": os.path.join(cache_path, "train_processed.arrow"),
            "test":  os.path.join(cache_path, "eval_processed.arrow"),
        },
        desc="Processing audio",
        num_proc=1,                 # Keep at 1 for Windows compatibility
    )

    # Tell HF dataset which columns are tensors
    hf_dataset.set_format(type="numpy", columns=["input_values", "attention_mask", "labels"], output_all_columns=True)

    print(f"   ✅ Processed → Train: {len(hf_dataset['train'])}, Eval: {len(hf_dataset['test'])}")

    # -------------------------
    # LOAD MODEL
    # -------------------------

    print("\n[4/6] Loading model...")

    model = AutoModelForCTC.from_pretrained(
        MODEL_NAME,
        ignore_mismatched_sizes=True,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.0,
        layerdrop=0.0,
    )

    model.to(DEVICE)
    model.config.ctc_zero_infinity  = True
    model.config.apply_spec_augment = False
    model.config.pad_token_id       = processor.tokenizer.pad_token_id
    model.freeze_feature_encoder()           # Saves VRAM, encoder already knows Hindi
    model.gradient_checkpointing_enable()    # Saves ~30% VRAM
    print(f"   ✅ Model loaded on {DEVICE}")

    # -------------------------
    # VERIFY
    # -------------------------

    print("\n[5/6] Verifying setup...")
    sample = hf_dataset["train"][0]
    print(f"   Text:            {sample['text'][:50]}")
    print(f"   input_values:    shape={np.array(sample['input_values']).shape}")
    print(f"   attention_mask:  shape={np.array(sample['attention_mask']).shape}")
    print(f"   labels:          {sample['labels'][:10]}")

    collator = DataCollatorCTC(processor, pad_id=processor.tokenizer.pad_token_id)
    batch    = collator([hf_dataset["train"][0], hf_dataset["train"][1]])
    print(f"   Batch input:     {batch['input_values'].shape}")
    print(f"   Batch labels:    {batch['labels'].shape}")
    print("   ✅ Collator works")

    # -------------------------
    # TRAIN
    # -------------------------

    print("\n[6/6] Training...")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR + "/full",

        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=8,      # Effective batch = 8

        learning_rate=3e-5,
        lr_scheduler_type="linear",
        warmup_steps=500,

        max_steps=4000,
        evaluation_strategy="steps",
        eval_steps=500,
        logging_steps=50,
        save_steps=1000,

        fp16=True,
        remove_unused_columns=False,
        dataloader_num_workers=2,           # Safe on Windows with preloaded data
        dataloader_pin_memory=True,
        max_grad_norm=1.0,
        report_to="none",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_dataset["train"],
        eval_dataset=hf_dataset["test"],
        tokenizer=processor.feature_extractor,
        data_collator=collator,
        callbacks=[InferenceCallback(processor, hf_dataset["test"], DEVICE)]
    )

    trainer.train()
    
    trainer.train()

    print("\n💾 Saving final model...")

    trainer.save_model(OUTPUT_DIR + "/final")
    processor.save_pretrained(OUTPUT_DIR + "/final")

    print("✅ Final model saved at:", OUTPUT_DIR + "/final")

    print("\n✅ Done!")
    