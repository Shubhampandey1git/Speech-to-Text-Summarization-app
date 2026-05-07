import pandas as pd
from transformers import pipeline
from tqdm import tqdm
from langdetect import detect
import torch
import os
import re

# ------------------------------------------
# Config
# ------------------------------------------
device = 0 if torch.cuda.is_available() else -1
USE_GEMMA = True
gemma = None  # 🔥 prevent reloading

# ------------------------------------------
# Load Models (ONLY ONCE)
# ------------------------------------------
def load_models(model_type="bart"):
    global gemma

    print("🔹 Loading models...")
    
    # GEMMA (Optional)
    if USE_GEMMA and gemma is None: 
        gemma = pipeline(
            "text-generation",
            model="google/gemma-2b",
            device=device,  # 🔥 GPU
            torch_dtype=torch.float16
        )

    # BART (Finetuned)
    if model_type == "bart":
        return pipeline(
            "summarization",
            model="models/summarization/english_bart",
            device=device
        )

    # PEGASUS
    elif model_type == "pegasus":
        return pipeline(
            "summarization",
            model="google/pegasus-xsum",
            device=device
        )

    # FlanT5
    elif model_type == "t5":
        return pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            device=device
        )

    # IndicBART
    elif model_type == "indicbart":
        return pipeline(
            "summarization",
            model="ai4bharat/IndicBART",
            device=device
        )

# ------------------------------------------
# Cleaning
# ------------------------------------------
def clean_text(text):
    if not USE_GEMMA:
        return text

    # 🔥 Short prompt for speed
    prompt = f"Fix ASR errors:\n{text}\nCleaned:"

    output = gemma(
        prompt,
        max_new_tokens=60,   # 🔥 smaller = faster
        do_sample=False      # 🔥 removes sampling overhead
    )[0]["generated_text"]

    # Extract only cleaned part
    cleaned = output.split("Cleaned:")[-1].strip()

    return cleaned if cleaned else text  # fallback

# ------------------------------------------
# Language Detection
# ------------------------------------------
def detect_language(text):
    try:
        return detect(text)
    except:
        return "en"

# ------------------------------------------
# Chunking
# ------------------------------------------
def chunk_text(text, max_words=200):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

# ------------------------------------------
# CSV Testing Function 🔥
# ------------------------------------------
def run_test(test_file, output_file, model_type):
    print(f"\n🔹 Loading {model_type} model...")
    
    summarizer = load_models(model_type)

    df = pd.read_csv(test_file)
    results = []

    for row in tqdm(df.itertuples(index=False), total=len(df)):
        dialogue = row.dialogue
        reference = row.summary

        # 🔥 Safety check
        if not dialogue or str(dialogue).strip() == "":
            results.append({
                "dialogue": dialogue,
                "reference_summary": reference,
                "generated_summary": ""
            })
            continue

        try:
            # 🔥 APPLY GEMMA CLEANING (only for longer inputs to save time)
            if len(dialogue.split()) > 30:
                cleaned = clean_text(dialogue)
            else:
                cleaned = dialogue

            # Optional: truncate very long inputs
            cleaned = cleaned[:1000]

            if model_type == "t5":
                dialogue_input = f"Summarize the following conversation:\n{cleaned}"

                prediction = summarizer(
                    dialogue_input,
                    max_length=100,
                    num_beams=4,
                    do_sample=False
                )[0]["generated_text"]

                # Clean special tokens (mT5/FLAN safety)
                prediction = re.sub(r"<extra_id_\d+>", "", prediction).strip()

            else:
                prediction = summarizer(
                    cleaned,
                    max_length=100,
                    min_length=30,
                    truncation=True
                )[0]["summary_text"]

            results.append({
                "dialogue": dialogue,
                "reference_summary": reference,
                "generated_summary": prediction
            })

        except Exception as e:
            print("Error:", e)
            results.append({
                "dialogue": dialogue,
                "reference_summary": reference,
                "generated_summary": ""
            })

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    pd.DataFrame(results).to_csv(output_file, index=False)

    print(f"✅ Saved: {output_file}")

    # 🔥 IMPORTANT: free memory
    del summarizer
    torch.cuda.empty_cache()

# ------------------------------------------
# MAIN
# ------------------------------------------
if __name__ == "__main__":
    TEST_FILE = "data/clean/samsum/clean_samsum_test.csv"
    
    # BATCH TESTING ENGLISH(Uncomment to run all at once, but beware of memory issues)
    
    # BART (Finetuned)
    run_test(TEST_FILE, "experiments/summarization/Gamma_bart_samsum/predictions.csv", "bart")
    
    # PEGASUS
    # run_test(TEST_FILE, "experiments/summarization/pegasus_samsum/predictions.csv", "pegasus")
    
    # FlanT5
    # run_test(TEST_FILE, "experiments/summarization/flanT5_samsum/predictions.csv", "t5")

    # IndicBART
    # run_test(TEST_FILE, "experiments/summarization/indicbart_samsum/predictions.csv", "indicbart")