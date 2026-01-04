from bert.finetune_bert_model import train_model
from data.get_data import load_data

tokenized_ds, _, mlb, label_names, metadata = load_data()

train_model(tokenized_ds, _, mlb, label_names, metadata)