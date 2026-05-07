from datasets import load_dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Trainer,
    TrainingArguments
)
import torch

# ------------------------------------------
# CONFIG
# ------------------------------------------
MODEL_TYPE = "indicbart"   # flant5 | indicbart
LANGUAGE = "hindi"

TRAIN_FILE = "data/clean/Hindi_text_summ/clean_Hinditxt_train.csv"
VAL_FILE = "data/clean/Hindi_text_summ/clean_Hinditxt_test.csv"

# ------------------------------------------
# MODEL SELECTION
# ------------------------------------------
MODELS = {
    "flant5": "google/flan-t5-base",
    "indicbart": "ai4bharat/IndicBART"
}

model_name = MODELS[MODEL_TYPE]

print(f"\n🔹 Loading Model: {model_name}")

# ------------------------------------------
# TOKENIZER + MODEL
# ------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32 if torch.cuda.is_available() else torch.float32
)

# ------------------------------------------
# LOAD DATASET
# ------------------------------------------
train_dataset = load_dataset(
    "csv",
    data_files=TRAIN_FILE
)["train"]

val_dataset = load_dataset(
    "csv",
    data_files=VAL_FILE
)["train"]

train_dataset = train_dataset.select(range(10000))
val_dataset = val_dataset.select(range(1000))

# ------------------------------------------
# STANDARDIZE COLUMNS
# ------------------------------------------
def standardize_columns(example):

    # Clean Hindi Dataset
    if "clean_dialogue" in example:
        dialogue = example["clean_dialogue"]
        summary = example["clean_summary"]

    # Generic Dataset
    elif "dialogue" in example:
        dialogue = example["dialogue"]
        summary = example["summary"]

    else:
        dialogue = ""
        summary = ""

    return {
        "dialogue": dialogue,
        "summary": summary
    }

# ------------------------------------------
# PREPROCESS
# ------------------------------------------
def preprocess(example):

    dialogues = example["dialogue"]

    # --------------------------------------
    # FLAN-T5 NEEDS PROMPT
    # --------------------------------------
    if MODEL_TYPE == "flant5":
        dialogues = [
            f"Summarize this Hindi conversation:\n{text}"
            for text in dialogues
        ]

    # --------------------------------------
    # INDICBART PREFIX
    # --------------------------------------
    elif MODEL_TYPE == "indicbart":
        dialogues = [
            f"summarize: {text}"
            for text in dialogues
        ]

    # --------------------------------------
    # TOKENIZE INPUTS
    # --------------------------------------
    inputs = tokenizer(
        dialogues,
        max_length=256,
        truncation=True,
        padding="max_length"
    )

    # --------------------------------------
    # TOKENIZE TARGETS
    # --------------------------------------
    targets = tokenizer(
        example["summary"],
        max_length=128,
        truncation=True,
        padding="max_length"
    )

    labels = targets["input_ids"]

    # Replace padding token id with -100
    labels = [
        [
            token if token != tokenizer.pad_token_id else -100
            for token in label
        ]
        for label in labels
    ]

    inputs["labels"] = labels

    # --------------------------------------
    # INDICBART FIX
    # --------------------------------------
    inputs.pop("token_type_ids", None)

    return inputs

# ------------------------------------------
# APPLY STANDARDIZATION
# ------------------------------------------
train_dataset = train_dataset.map(standardize_columns)
val_dataset = val_dataset.map(standardize_columns)

dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})

# ------------------------------------------
# REMOVE UNUSED COLUMNS
# ------------------------------------------
remove_cols = [
    col for col in ["id", "dialogue", "summary"]
    if col in dataset["train"].column_names
]

dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=remove_cols
)

print("\n✅ Sample:")
print(dataset["train"][0])

# ------------------------------------------
# TRAINING ARGUMENTS
# ------------------------------------------
training_args = TrainingArguments(
    output_dir=f"models/summarization/{LANGUAGE}_{MODEL_TYPE}",

    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,

    gradient_accumulation_steps=4,

    learning_rate=5e-5,
    warmup_steps=500,

    num_train_epochs=3,

    evaluation_strategy="epoch",
    save_strategy="epoch",

    logging_dir="./logs",

    dataloader_num_workers=0,
    dataloader_pin_memory=True,

    save_total_limit=2,

    fp16=False,

    load_best_model_at_end=True,

    report_to="none"
)

# ------------------------------------------
# TRAINER
# ------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"]
)

# ------------------------------------------
# MAIN
# ------------------------------------------
if __name__ == "__main__":

    trainer.train()

    save_path = f"models/summarization/{LANGUAGE}_{MODEL_TYPE}"

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    print(f"\n✅ Model saved to: {save_path}")