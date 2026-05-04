import pandas as pd
import json
import re

HINDI_CSV = "data/processed/asr/hindi/train.csv"
# ENGLISH_CSV = "data/processed/asr/english/train.csv"


def clean_text(text):
    text = str(text)

    # remove English letters
    text = re.sub(r"[A-Za-z]", "", text)

    # remove digits
    text = re.sub(r"[0-9]", "", text)

    # remove punctuation (keep Hindi danda)
    text = re.sub(r"[#,'\-.:\\[\]]", "", text)

    # remove zero-width characters
    text = text.replace("\u200b", "").replace("\u200c", "")

    # normalize spaces
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\u0900-\u097F| ]", "", text)

    return text.strip()

def extract_vocab():
    df = pd.read_csv(HINDI_CSV)

    # merge all text
    df["text"] = df["text"].apply(clean_text)

    all_text = " ".join(df["text"].tolist())

    # normalize spaces
    all_text = re.sub(r"\s+", " ", all_text)

    # replace space with '|'
    all_text = all_text.replace(" ", "|")

    # get unique characters
    vocab = sorted(list(set(all_text)))

    # build vocab dict
    vocab_dict = {v: i for i, v in enumerate(vocab)}

    # add special tokens (ONLY ONCE)
    vocab_dict["[UNK]"] = len(vocab_dict)
    vocab_dict["[PAD]"] = len(vocab_dict)

    # save
    with open("models/asr/hindi/vocab.json", "w", encoding="utf-8") as f:
        json.dump(vocab_dict, f, ensure_ascii=False, indent=2)

    print("Vocab size:", len(vocab_dict))


if __name__ == "__main__":
    extract_vocab()