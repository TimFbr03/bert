from data.get_data import load_data
from bert.finetune_bert_model import train_model


def main():
    tokenized_datasets, tokenizer, metadata = load_data()
    train_model(tokenized_datasets, metadata)


if __name__ == "__main__":
    main()
