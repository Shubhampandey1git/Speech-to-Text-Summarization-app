from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
import os

# ------------------------------------------
# SAVE PATHS
# ------------------------------------------
FLANT5_SAVE_PATH = "models/summarization/hindi_flant5_pretrained"
INDICBART_SAVE_PATH = "models/summarization/hindi_indicbart_pretrained"

# ------------------------------------------
# CREATE DIRECTORIES
# ------------------------------------------
os.makedirs(FLANT5_SAVE_PATH, exist_ok=True)
os.makedirs(INDICBART_SAVE_PATH, exist_ok=True)

# ------------------------------------------
# DOWNLOAD FLAN-T5
# ------------------------------------------
print("\n🔹 Downloading FLAN-T5...")

flant5_model_name = "google/flan-t5-base"

flant5_tokenizer = AutoTokenizer.from_pretrained(
    flant5_model_name
)

flant5_model = AutoModelForSeq2SeqLM.from_pretrained(
    flant5_model_name
)

flant5_tokenizer.save_pretrained(
    FLANT5_SAVE_PATH
)

flant5_model.save_pretrained(
    FLANT5_SAVE_PATH
)

print(f"✅ FLAN-T5 saved to: {FLANT5_SAVE_PATH}")

# ------------------------------------------
# DOWNLOAD INDICBART
# ------------------------------------------
print("\n🔹 Downloading IndicBART...")

indicbart_model_name = "ai4bharat/IndicBART"

indicbart_tokenizer = AutoTokenizer.from_pretrained(
    indicbart_model_name
)

indicbart_model = AutoModelForSeq2SeqLM.from_pretrained(
    indicbart_model_name
)

indicbart_tokenizer.save_pretrained(
    INDICBART_SAVE_PATH
)

indicbart_model.save_pretrained(
    INDICBART_SAVE_PATH
)

print(f"✅ IndicBART saved to: {INDICBART_SAVE_PATH}")

print("\n🎉 All models downloaded successfully.")