from datasets import load_dataset, Dataset, DatasetDict
from transformers import RobertaTokenizer
from sklearn.model_selection import train_test_split
import pandas as pd

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

# --------------------------------------------------
# Load and filter raw data
# --------------------------------------------------
def load_raw_dataframe():
    hf_ds = load_dataset("Tobi-Bueck/customer-support-tickets")
    df = hf_ds["train"].to_pandas()

    df = df[df["queue"].isin(IT_QUEUES)]
    df = df[["subject", "body", "type", "queue", "priority"]]

    # Collapse IT & Technology subqueues
    df["queue"] = df["queue"].replace(
        to_replace=r"^IT & Technology/.*",
        value="IT & Technology",
        regex=True,
    )

    # Drop rare priority classes
    df = df[~df["priority"].isin(DROP_PRIORITIES)].reset_index(drop=True)

    return df


# --------------------------------------------------
# Text features
# --------------------------------------------------
def create_text_features(df):
    df = df.copy()
    df["text"] = (
        df["subject"].fillna("") + " [SEP] " + df["body"].fillna("")
    )
    return df


# --------------------------------------------------
# Label encoding (per head)
# --------------------------------------------------
def encode_labels(df):
    df = df.copy()
    label_encoders = {}

    for col in ["type", "queue", "priority"]:
        classes = sorted(df[col].unique())
        label_encoders[col] = {c: i for i, c in enumerate(classes)}
        df[col] = df[col].map(label_encoders[col])

    return df, label_encoders


# --------------------------------------------------
# Train / validation / test split
# --------------------------------------------------
def split_dataset(df, test_size=0.2, val_size=0.1):
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size,
        random_state=RANDOM_STATE,
        shuffle=True,
    )

    return train_df, val_df, test_df


# --------------------------------------------------
# Tokenization
# --------------------------------------------------
def tokenize_dataset(train_df, val_df, test_df):
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    keep_cols = ["text", "type", "queue", "priority"]

    train_ds = Dataset.from_pandas(train_df[keep_cols].reset_index(drop=True))
    val_ds = Dataset.from_pandas(val_df[keep_cols].reset_index(drop=True))
    test_ds = Dataset.from_pandas(test_df[keep_cols].reset_index(drop=True))

    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
        )

        tokenized["labels"] = {
            "type": examples["type"],
            "queue": examples["queue"],
            "priority": examples["priority"],
        }

        return tokenized

    train_ds = train_ds.map(tokenize_function, batched=True, remove_columns=keep_cols)
    val_ds = val_ds.map(tokenize_function, batched=True, remove_columns=keep_cols)
    test_ds = test_ds.map(tokenize_function, batched=True, remove_columns=keep_cols)

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
    df = create_text_features(df)
    df, label_encoders = encode_labels(df)

    train_df, val_df, test_df = split_dataset(df)

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
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
    }

    print("Dataset prepared successfully")
    print(f"Train: {metadata['train_size']}")
    print(f"Validation: {metadata['val_size']}")
    print(f"Test: {metadata['test_size']}")
    print(f"Type classes: {metadata['num_type']}")
    print(f"Queue classes: {metadata['num_queue']}")
    print(f"Priority classes: {metadata['num_priority']}")

    return tokenized_datasets, tokenizer, metadata


# --------------------------------------------------
# Debug / standalone execution
# --------------------------------------------------
if __name__ == "__main__":
    tokenized_datasets, tokenizer, metadata = load_data()

    sample = tokenized_datasets["train"][0]
    print("\nSample labels:")
    print(sample["labels"])
    print("Input length:", len(sample["input_ids"]))
