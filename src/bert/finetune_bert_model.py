import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoModel, get_linear_schedule_with_warmup
import mlflow
import mlflow.pytorch
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from collections import defaultdict
import numpy as np

from data.get_data import load_data

# Config
BATCH_SIZE = 8
EPOCHS = 4
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EXPERIMENT_NAME = "" # Link einfügen

mlflow.set_experiment(EXPERIMENT_NAME)

# Loss scaling
LOSS_WEIGHTS = {
    "type": 1.0,
    "queue": 1.5,
    "priority": 1.0,
}


class RobertaMultiHead(nn.Module):
    def __init__(self, model_name, num_type, num_queue, num_priority):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size

        self.type_head = nn.Linear(hidden, num_type)
        self.queue_head = nn.Linear(hidden, num_queue)
        self.priority_head = nn.Linear(hidden, num_priority)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        cls = outputs.last_hidden_state[:, 0]

        return {
            "type": self.type_head(cls),
            "queue": self.queue_head(cls),
            "priority": self.priority_head(cls),
        }


# Utils
def compute_class_weights(class_counts: dict):
    weights = {}
    for head, counts in class_counts.items():
        total = sum(counts.values())
        num_classes = len(counts)
        weights[head] = torch.tensor(
            [total / (num_classes * counts[i]) for i in range(num_classes)],
            dtype=torch.float,
            device=DEVICE,
        )
    return weights


def collate_fn(batch):
    return {
        "input_ids": torch.stack([x["input_ids"] for x in batch]),
        "attention_mask": torch.stack([x["attention_mask"] for x in batch]),
        "labels": {
            "type": torch.tensor([x["type_label"] for x in batch]),
            "queue": torch.tensor([x["queue_label"] for x in batch]),
            "priority": torch.tensor([x["priority_label"] for x in batch]),
        },
    }


def evaluate(model, dataloader, device, label_names):
    model.eval()
    preds = defaultdict(list)
    labels = defaultdict(list)

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            batch_labels = batch["labels"]

            outputs = model(input_ids, attention_mask)

            for head in outputs:
                preds[head].extend(outputs[head].argmax(dim=1).cpu().numpy())
                labels[head].extend(batch_labels[head].numpy())

    metrics = {}

    for head in preds:
        report = classification_report(
            labels[head],
            preds[head],
            output_dict=True,
            zero_division=0,
        )

        metrics[f"{head}_macro_f1"] = report["macro avg"]["f1-score"]

        # per-class recall & precision
        for class_id, stats in report.items():
            if class_id.isdigit():
                class_name = label_names[head][int(class_id)]
                metrics[f"{head}_recall/{class_name}"] = stats["recall"]
                metrics[f"{head}_precision/{class_name}"] = stats["precision"]
                metrics[f"{head}_support/{class_name}"] = stats["support"]

    return metrics

# Training loop
def train():
    with mlflow.start_run():

        datasets, tokenizer, metadata = load_data()

        mlflow.log_params({
            "model": metadata["model_name"],
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "warmup_ratio": WARMUP_RATIO,
            "max_length": metadata["max_length"],
        })

        train_loader = DataLoader(
            datasets["train"],
            batch_size=BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            datasets["validation"],
            batch_size=BATCH_SIZE,
            shuffle=False,
            collate_fn=collate_fn,
        )

        model = RobertaMultiHead(
            model_name=metadata["model_name"],
            num_type=metadata["num_type"],
            num_queue=metadata["num_queue"],
            num_priority=metadata["num_priority"],
        ).to(DEVICE)

        class_weights = compute_class_weights(metadata["class_counts"])

        losses = {
            head: nn.CrossEntropyLoss(weight=class_weights[head])
            for head in class_weights
        }

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        total_steps = len(train_loader) * EPOCHS
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(WARMUP_RATIO * total_steps),
            num_training_steps=total_steps,
        )

        for epoch in range(EPOCHS):
            model.train()
            epoch_loss = 0.0

            for batch in train_loader:
                optimizer.zero_grad()

                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)
                labels = batch["labels"]

                outputs = model(input_ids, attention_mask)

                loss = 0.0
                for head in outputs:
                    head_loss = losses[head](
                        outputs[head],
                        labels[head].to(DEVICE),
                    )
                    loss += LOSS_WEIGHTS[head] * head_loss

                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            val_metrics = evaluate(model, val_loader, DEVICE)

            mlflow.log_metric("train_loss", avg_loss, step=epoch)
            for k, v in val_metrics.items():
                mlflow.log_metric(f"val_{k}", v, step=epoch)

            print(
                f"Epoch {epoch + 1}/{EPOCHS} | "
                f"Loss: {avg_loss:.4f} | "
                + " | ".join(f"{k}: {v:.4f}" for k, v in val_metrics.items())
            )

        mlflow.pytorch.log_model(model, "model")

if __name__ == "__main__":
    train()
