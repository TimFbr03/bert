from datasets import load_dataset, Dataset, DatasetDict
from transformers import RobertaTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import mlflow
import mlflow.data

# --------------------------------------------------
# Configuration
# --------------------------------------------------
MODEL_NAME = "roberta-base"
MAX_LENGTH = 512
RANDOM_STATE = 42

IT_QUEUES = [
    "Technical Support",
    "IT Support",
    "Product Support",
    "IT & Technology/Hardware Support",
    "IT & Technology/Software Development",
    "IT & Technology/Security Operations",
    "IT & Technology/Network Infrastructure",
    "Service Outages and Maintenance",
]

DROP_PRIORITIES = {"critical", "very_low"}

LABEL_COLUMNS = ["type", "queue", "priority"]
TEXT_COLUMNS = ["subject", "body"]

# --------------------------------------------------
# Load and filter raw data
# --------------------------------------------------
def load_raw_dataframe() -> pd.DataFrame:
    hf_ds = load_dataset("Tobi-Bueck/customer-support-tickets")
    df = hf_ds["train"].to_pandas()

    # Keep relevant queues
    df = df[df["queue"].isin(IT_QUEUES)]

    # Select required columns
    df = df[TEXT_COLUMNS + LABEL_COLUMNS]

    # Collapse IT & Technology subqueues
    df["queue"] = df["queue"].replace(
        to_replace=r"^IT & Technology/.*",
        value="IT & Technology",
        regex=True,
    )

    # Drop rare priorities
    df = df[~df["priority"].isin(DROP_PRIORITIES)]

    # Drop rows with missing labels
    df = df.dropna(subset=LABEL_COLUMNS).reset_index(drop=True)

    return df


# --------------------------------------------------
# Label encoding (per head)
# --------------------------------------------------
def encode_labels(df: pd.DataFrame):
    df = df.copy()

    label_encoders = {}
    inverse_label_encoders = {}

    for col in LABEL_COLUMNS:
        classes = sorted(df[col].unique())
        encoder = {c: i for i, c in enumerate(classes)}
        inverse_encoder = {i: c for c, i in encoder.items()}

        df[col] = df[col].map(encoder)

        label_encoders[col] = encoder
        inverse_label_encoders[col] = inverse_encoder

    return df, label_encoders, inverse_label_encoders


# --------------------------------------------------
# Class counts (for imbalance handling)
# --------------------------------------------------
def compute_class_counts(df: pd.DataFrame):
    return {
        col: df[col].value_counts().sort_index().to_dict()
        for col in LABEL_COLUMNS
    }


# --------------------------------------------------
# Train / validation / test split (stratified)
# --------------------------------------------------
def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float = 0.1,
):
    # Stratify on most imbalanced / critical head: queue
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=df["queue"],
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=RANDOM_STATE,
        stratify=train_val_df["queue"],
    )

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


# --------------------------------------------------
# Tokenization
# --------------------------------------------------
def tokenize_dataset(train_df, val_df, test_df):
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    keep_cols = TEXT_COLUMNS + LABEL_COLUMNS

    train_ds = Dataset.from_pandas(train_df[keep_cols])
    val_ds = Dataset.from_pandas(val_df[keep_cols])
    test_ds = Dataset.from_pandas(test_df[keep_cols])

    def tokenize_function(examples):
        # --- robust text sanitation ---
        subjects = [
            s if isinstance(s, str) else ""
            for s in examples["subject"]
        ]
        bodies = [
            b if isinstance(b, str) else ""
            for b in examples["body"]
        ]

        tokenized = tokenizer(
            subjects,
            bodies,
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

        # --- FLAT label columns (REQUIRED) ---
        tokenized["type_label"] = examples["type"]
        tokenized["queue_label"] = examples["queue"]
        tokenized["priority_label"] = examples["priority"]

        return tokenized

    train_ds = train_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=keep_cols,
    )
    val_ds = val_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=keep_cols,
    )
    test_ds = test_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=keep_cols,
    )

    # --- explicit torch formatting ---
    for ds in (train_ds, val_ds, test_ds):
        ds.set_format(
            type="torch",
            columns=[
                "input_ids",
                "attention_mask",
                "type_label",
                "queue_label",
                "priority_label",
            ],
        )

    return DatasetDict(
        train=train_ds,
        validation=val_ds,
        test=test_ds,
    ), tokenizer

# --------------------------------------------------
# Public entry point
# --------------------------------------------------
def load_data():
    df = load_raw_dataframe()

    df, label_encoders, inverse_label_encoders = encode_labels(df)
    class_counts = compute_class_counts(df)

    train_df, val_df, test_df = split_dataset(df)

    # --------------------------------------------------
    # MLflow Dataset logging (PRE-TOKENIZATION)
    # --------------------------------------------------
    if mlflow.active_run() is not None:
        for split_name, split_df in {
            "train": train_df,
            "validation": val_df,
            "test": test_df,
        }.items():
            dataset = mlflow.data.from_pandas(
                split_df,
                name="customer-support-tickets-it",
                source="hf://Tobi-Bueck/customer-support-tickets",
            )
            mlflow.log_input(dataset, context=split_name)


    tokenized_datasets, tokenizer = tokenize_dataset(
        train_df, val_df, test_df
    )

    metadata = {
        "model_name": MODEL_NAME,
        "max_length": MAX_LENGTH,
        "num_type": len(label_encoders["type"]),
        "num_queue": len(label_encoders["queue"]),
        "num_priority": len(label_encoders["priority"]),
        "label_encoders": label_encoders,
        "inverse_label_encoders": inverse_label_encoders,
        "class_counts": class_counts,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "random_state": RANDOM_STATE,
    }

    print("Dataset prepared successfully")
    print(f"Train size: {metadata['train_size']}")
    print(f"Validation size: {metadata['val_size']}")
    print(f"Test size: {metadata['test_size']}")
    print(f"Type classes: {metadata['num_type']}")
    print(f"Queue classes: {metadata['num_queue']}")
    print(f"Priority classes: {metadata['num_priority']}")

    return tokenized_datasets, tokenizer, metadata


# --------------------------------------------------
# Debug / standalone execution
# --------------------------------------------------
if __name__ == "__main__":
    datasets, tokenizer, metadata = load_data()

    # sample = datasets["train"][0]
    # print("\nSample labels:")
    # print(sample["labels"])
    # print("Input length:", sample["input_ids"].shape[0])