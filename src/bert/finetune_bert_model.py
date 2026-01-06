import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch

from sklearn.metrics import f1_score

from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# --------------------------------------------------
# Multi-head model
# --------------------------------------------------
class MultiHeadXLMRoberta(nn.Module):
    def __init__(self, model_name, num_type, num_queue, num_priority):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(model_name)
        hidden_size = self.encoder.config.hidden_size

        self.dropout = nn.Dropout(0.1)

        self.type_head = nn.Linear(hidden_size, num_type)
        self.queue_head = nn.Linear(hidden_size, num_queue)
        self.priority_head = nn.Linear(hidden_size, num_priority)

        self.loss_fct = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        pooled = outputs.last_hidden_state[:, 0]
        pooled = self.dropout(pooled)

        logits = {
            "type": self.type_head(pooled),
            "queue": self.queue_head(pooled),
            "priority": self.priority_head(pooled),
        }

        loss = None
        if labels is not None:
            loss = (
                self.loss_fct(logits["type"], labels["type"])
                + self.loss_fct(logits["queue"], labels["queue"])
                + self.loss_fct(logits["priority"], labels["priority"])
            )

        return {
            "loss": loss,
            "logits": logits,
        }


# --------------------------------------------------
# Custom Trainer
# --------------------------------------------------
class MultiHeadTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs, labels=labels)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss


# --------------------------------------------------
# Metrics (per head)
# --------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    metrics = {}

    for head in ["type", "queue", "priority"]:
        preds = logits[head].argmax(axis=1)
        metrics[f"{head}_f1_macro"] = f1_score(
            labels[head],
            preds,
            average="macro",
            zero_division=0,
        )

    metrics["f1_macro_mean"] = sum(
        metrics[f"{h}_f1_macro"] for h in ["type", "queue", "priority"]
    ) / 3

    return metrics


# --------------------------------------------------
# Training entry point
# --------------------------------------------------
def train_model(tokenized_ds, metadata):
    model_name = "FacebookAI/xlm-roberta-base"
    output_dir = "./xlm-roberta-customer-support"
    final_model_dir = f"{output_dir}/final"

    mlflow.set_experiment("xlm-roberta-multihead-customer-support")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = MultiHeadXLMRoberta(
        model_name=model_name,
        num_type=metadata["num_type"],
        num_queue=metadata["num_queue"],
        num_priority=metadata["num_priority"],
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_dir="./logs",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro_mean",
        save_total_limit=2,
        fp16=True,
        report_to="none",
    )

    trainer = MultiHeadTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    with mlflow.start_run():

        # -------- Params --------
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("num_type", metadata["num_type"])
        mlflow.log_param("num_queue", metadata["num_queue"])
        mlflow.log_param("num_priority", metadata["num_priority"])
        mlflow.log_param("learning_rate", training_args.learning_rate)
        mlflow.log_param("train_batch_size", training_args.per_device_train_batch_size)
        mlflow.log_param("epochs", training_args.num_train_epochs)

        # -------- Training --------
        trainer.train()

        # -------- Evaluation --------
        eval_metrics = trainer.evaluate()
        for k, v in eval_metrics.items():
            if isinstance(v, (float, int)):
                mlflow.log_metric(k, v)

        # -------- Save --------
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)

        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
        )


# --------------------------------------------------
# Guard
# --------------------------------------------------
if __name__ == "__main__":
    raise RuntimeError(
        "train_model(...) must be called from the pipeline with prepared datasets."
    )