import pandas as pd
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)
from tqdm import tqdm
import torch
import os
import re

# ------------------------------------------
# Config
# ------------------------------------------
INDICBART_MODEL = "models/summarization/hindi_indicbart_pretrained"
FLANT5_MODEL = "models/summarization/hindi_flant5_pretrained"
device = 0 if torch.cuda.is_available() else -1

# ------------------------------------------
# Load Models (ONLY ONCE)
# ------------------------------------------
def load_models(model_type="flant5"):

    print(f"\n🔹 Loading {model_type} model...")

    # --------------------------------------
    # FLAN-T5
    # --------------------------------------
    if model_type == "flant5":

        return pipeline(
            "text2text-generation",
            model=FLANT5_MODEL,
            device=device
        )

    # --------------------------------------
    # INDICBART
    # --------------------------------------
    elif model_type == "indicbart":

        tokenizer = AutoTokenizer.from_pretrained(
            INDICBART_MODEL
        )

        model = AutoModelForSeq2SeqLM.from_pretrained(
            INDICBART_MODEL,
            torch_dtype=torch.float16
        )

        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()

        return tokenizer, model

# ------------------------------------------
# FLAN-T5 SUMMARIZATION
# ------------------------------------------
def flant5_summary(summarizer, texts, batch_size):

    prompts = [f"""
    Summarize the following Hindi conversation:

    {text}
    """ for text in texts]
    with torch.no_grad():
        outputs = summarizer(
            prompts,
            max_length=60,
            num_beams=2,
            do_sample=False,
            truncation=True,
            batch_size=batch_size
        )

    # Remove special tokens if any
    predictions = [
                output["generated_text"].strip()
                for output in outputs
            ]

    return predictions

# ------------------------------------------
# INDICBART SUMMARIZATION
# ------------------------------------------
def indicbart_summary(tokenizer, model, texts):

    device_name = "cuda" if torch.cuda.is_available() else "cpu"

    # IMPORTANT LANGUAGE TAG
    input_texts = [
        f"<2hi> summarize: {text}"
        for text in texts
    ]

    # --------------------------------------
    # TOKENIZATION
    # --------------------------------------
    inputs = tokenizer(
        input_texts,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    # Remove token_type_ids
    inputs.pop("token_type_ids", None)

    # Move to GPU
    inputs = {
        k: v.to(device_name)
        for k, v in inputs.items()
    }

    # --------------------------------------
    # GENERATION
    # --------------------------------------
    with torch.no_grad():

        summary_ids = model.generate(
            **inputs,
            max_length=60,
            min_length=15,
            num_beams=1,
            repetition_penalty=1.8,
            early_stopping=True
        )

    # --------------------------------------
    # DECODE
    # --------------------------------------
    summaries = tokenizer.batch_decode(
        summary_ids,
        skip_special_tokens=True
    )

    # Cleanup
    summaries = [
        summary.replace("<2hi>", "")
            .replace("summarize:", "")
            .strip()
        for summary in summaries
    ]

    return summaries

# ------------------------------------------
# CSV Testing Function 🔥
# ------------------------------------------
def run_test(test_file, output_file, model_type):

    print(f"\n🔹 Running inference using {model_type}")

    df = pd.read_csv(test_file)

    results = []

    # --------------------------------------
    # LOAD MODEL
    # --------------------------------------
    if model_type == "flant5":
        summarizer = load_models("flant5")

    elif model_type == "indicbart":
        tokenizer, model = load_models("indicbart")

    # --------------------------------------
    # INFERENCE LOOP
    # --------------------------------------
    BATCH_SIZE = 16
    for i in tqdm(range(0, len(df), BATCH_SIZE)):

        batch_df = df.iloc[i:i+BATCH_SIZE]

        dialogues = batch_df["article"].fillna("").astype(str).tolist()
        references = batch_df["headline"].fillna("").astype(str).tolist()

        try:

            # ----------------------------------
            # FLAN-T5
            # ----------------------------------
            if model_type == "flant5":

                predictions = flant5_summary(
                    summarizer,
                    dialogues,
                    BATCH_SIZE
                )

            # ----------------------------------
            # INDICBART
            # ----------------------------------
            elif model_type == "indicbart":
                
                predictions = indicbart_summary(
                    tokenizer,
                    model,
                    dialogues
                )

            else:
                predictions = [""] * len(dialogues)

            # ----------------------------------
            # SAVE RESULTS
            # ----------------------------------

            for dialogue, reference, prediction in zip(
                    dialogues,
                    references,
                    predictions
                ):

                    results.append({
                        "dialogue": dialogue,
                        "reference_summary": reference,
                        "generated_summary": prediction
                    })

        except Exception as e:

            print("Error:", e)

            for dialogue, reference in zip(dialogues, references):
                results.append({
                    "dialogue": dialogue,
                    "reference_summary": reference,
                    "generated_summary": ""
                })

    # --------------------------------------
    # SAVE CSV
    # --------------------------------------
    os.makedirs(
        os.path.dirname(output_file),
        exist_ok=True
    )

    pd.DataFrame(results).to_csv(
        output_file,
        index=False
    )

    print(f"\n✅ Saved: {output_file}")

    # --------------------------------------
    # FREE GPU MEMORY
    # --------------------------------------
    torch.cuda.empty_cache()

# ------------------------------------------
# MAIN
# ------------------------------------------
if __name__ == "__main__":

    TEST_FILE = "data/clean/Hindi_text_summ/clean_Hinditxt_test.csv"

    # --------------------------------------
    # FLAN-T5
    # --------------------------------------
    run_test(
        TEST_FILE,
        "experiments/summarization/flant5_hindi_pretrained/predictions.csv",
        "flant5"
    )

    # --------------------------------------
    # INDICBART
    # --------------------------------------
    run_test(
        TEST_FILE,
        "experiments/summarization/indicbart_hindi_pretrained/predictions.csv",
        "indicbart"
    )