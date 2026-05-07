from datasets import load_dataset, DatasetDict
from transformers import BartTokenizer, BartForConditionalGeneration, Trainer, TrainingArguments

model_name = "facebook/bart-base"

tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Load CSV dataset
train_dataset = load_dataset(
    "csv",
    data_files="data/clean/samsum/clean_samsum_train.csv"
)["train"]

val_dataset = load_dataset(
    "csv",
    data_files="data/clean/samsum/clean_samsum_val.csv"
)["train"]

def standardize_columns(example):
    # Case 1: already clean
    if "clean_dialogue" in example:
        example["dialogue"] = example["clean_dialogue"]
        example["summary"] = example["clean_summary"]

    # Case 2: original SAMSum
    elif "dialogue" in example:
        example["dialogue"] = example["dialogue"]
        example["summary"] = example["summary"]

    return {
        "dialogue": example["dialogue"],
        "summary": example["summary"]
    }

def preprocess(example):
    inputs = tokenizer(
        example["dialogue"],
        max_length=256,
        truncation=True,
        padding="max_length"
    )

    targets = tokenizer(
        example["summary"],
        max_length=64,
        truncation=True,
        padding="max_length"
    )

    labels = targets["input_ids"]

    # Replace padding token id with -100
    labels = [
        [(token if token != tokenizer.pad_token_id else -100) for token in label]
        for label in labels
    ]

    inputs["labels"] = labels
    return inputs

train_dataset = train_dataset.map(standardize_columns)
val_dataset = val_dataset.map(standardize_columns)

dataset = DatasetDict({
    "train": train_dataset,
    "validation": val_dataset
})

remove_cols = [col for col in ["id", "dialogue", "summary"] if col in dataset["train"].column_names]

dataset = dataset.map(
    preprocess,
    batched=True,
    remove_columns=remove_cols
)

print("Sample:", dataset["train"][0])

# Training arguments

training_args = TrainingArguments(
    output_dir="models/summarization/english",
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    warmup_steps=500,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    logging_dir="./logs",
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    save_total_limit=2,
    fp16=True,
    
    load_best_model_at_end=True,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"]
)

if __name__ == "__main__":
    trainer.train()

    model.save_pretrained("models/summarization/english")
    tokenizer.save_pretrained("models/summarization/english")