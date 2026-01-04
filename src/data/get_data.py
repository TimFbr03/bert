from datasets import load_dataset, DatasetDict, Dataset
from transformers import RobertaTokenizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np

# Load Hugging Face dataset and convert to pandas
hf_ds = load_dataset("Tobi-Bueck/customer-support-tickets")
df = hf_ds["train"].to_pandas()

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

df = df[df["queue"].isin(IT_QUEUES)].reset_index(drop=True)
df = df[["subject", "body", "type", "queue", "priority"]]

def prepare_multi_label_data(df, label_columns=["type", "queue", "priority"]):
    # Combine all label columns into tuples
    df["labels"] = df[label_columns].apply(
        lambda row: tuple(f"{col}:{val}" for col, val in row.items() if pd.notna(val)), 
        axis=1
    )
    
    # Convert tuples to lists for MultiLabelBinarizer
    labels_list = df["labels"].apply(list).tolist()
    
    # Fit MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    label_matrix = mlb.fit_transform(labels_list)
    
    # Create processed dataframe
    df_processed = df.copy()
    df_processed["label_vector"] = list(label_matrix)
    
    return df_processed, mlb, mlb.classes_.tolist()

def create_text_features(df, text_columns=["subject", "body"]):
    df = df.copy()
    df["text"] = df[text_columns].apply(
        lambda row: " [SEP] ".join([str(val) for val in row if pd.notna(val)]), 
        axis=1
    )
    return df

def split_dataset(df, test_size=0.2, val_size=0.1, random_state=42):
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        random_state=random_state,
        shuffle=True
    )
    
    # Second split: separate validation from training
    train_df, val_df = train_test_split(
        train_val_df, 
        test_size=val_size, 
        random_state=random_state,
        shuffle=True
    )
    
    return train_df, val_df, test_df

def tokenize_dataset(train_df, val_df, test_df, model_name="roberta-base", max_length=512):
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    
    # Convert dataframes to Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df[["text", "label_vector"]].reset_index(drop=True))
    val_dataset = Dataset.from_pandas(val_df[["text", "label_vector"]].reset_index(drop=True))
    test_dataset = Dataset.from_pandas(test_df[["text", "label_vector"]].reset_index(drop=True))
    
    def tokenize_function(examples):
        tokenized = tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

        tokenized["labels"] = [
            [float(x) for x in label_vec]
            for label_vec in examples["label_vector"]
        ]

        return tokenized

    # Apply tokenization
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_val = val_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
    
    # Create DatasetDict
    tokenized_datasets = DatasetDict({
        "train": tokenized_train,
        "validation": tokenized_val,
        "test": tokenized_test
    })
    
    return tokenized_datasets, tokenizer

def load_data():
    # Prepare multi-label data
    df_processed, mlb, label_names = prepare_multi_label_data(df)
    
    # Create combined text features
    df_processed = create_text_features(df_processed)
    
    # Split dataset
    train_df, val_df, test_df = split_dataset(df_processed)
    
    # Tokenize datasets
    tokenized_datasets, tokenizer = tokenize_dataset(train_df, val_df, test_df)
    
    # Create metadata
    metadata = {
        "num_labels": len(label_names),
        "label_names": label_names,
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "max_length": 512,
        "model_name": "roberta-base"
    }
    
    print(f"Dataset prepared successfully!")
    print(f"Train size: {metadata['train_size']}")
    print(f"Validation size: {metadata['val_size']}")
    print(f"Test size: {metadata['test_size']}")
    print(f"Number of labels: {metadata['num_labels']}")
    print(f"Labels: {label_names[:5]}...")  # Show first 5 labels
    
    return tokenized_datasets, tokenizer, mlb, label_names, metadata

# Example usage
if __name__ == "__main__":
    tokenized_datasets, tokenizer, mlb, label_names, metadata = load_data()
    
    # Access individual splits
    train = tokenized_datasets["train"]
    validation = tokenized_datasets["validation"]
    test = tokenized_datasets["test"]
    
    # Example: View first sample
    print("\nFirst training example:")
    print(f"Input IDs shape: {len(train[0]['input_ids'])}")
    print(f"Text: {train[0]['text']}")
    print(f"Labels: {train[0]['labels']}")
    print(f"Number of active labels: {sum(train[0]['labels'])}")