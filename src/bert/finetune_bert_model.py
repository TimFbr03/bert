import numpy as np
import torch
import mlflow
import mlflow.pytorch

from sklearn.metrics import f1_score, precision_score, recall_score

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# --------------------------------------------------
# Metrics for multi-label classification
# --------------------------------------------------
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    probs = torch.sigmoid(torch.tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)

    return {
        "f1_micro": f1_score(labels, preds, average="micro", zero_division=0),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "precision_micro": precision_score(labels, preds, average="micro", zero_division=0),
        "recall_micro": recall_score(labels, preds, average="micro", zero_division=0),
    }


def train_model(tokenized_ds, mlb, label_names, metadata):
    # --------------------------------------------------
    # Configuration
    # --------------------------------------------------
    model_name = "FacebookAI/xlm-roberta-base"
    output_dir = "./xlm-roberta-customer-support"
    final_model_dir = f"{output_dir}/final"

    num_labels = metadata["num_labels"]

    label2id = {label: i for i, label in enumerate(label_names)}
    id2label = {i: label for label, i in label2id.items()}

    # --------------------------------------------------
    # MLflow setup
    # --------------------------------------------------
    mlflow.set_experiment("xlm-roberta-multilabel-customer-support")

    # --------------------------------------------------
    # Tokenizer & model
    # --------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        problem_type="multi_label_classification",
    )

    data_collator = DataCollatorWithPadding(tokenizer)

    # --------------------------------------------------
    # Training arguments
    # --------------------------------------------------
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
        metric_for_best_model="f1_micro",
        save_total_limit=2,
        fp16=True,
        report_to="none",  # IMPORTANT: manual MLflow logging
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_ds["train"],
        eval_dataset=tokenized_ds.get("validation") or tokenized_ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # --------------------------------------------------
    # MLflow run
    # --------------------------------------------------
    with mlflow.start_run():

        # -------- Parameters --------
        mlflow.log_param("model_name", model_name)
        mlflow.log_param("num_labels", num_labels)
        mlflow.log_param("learning_rate", training_args.learning_rate)
        mlflow.log_param("train_batch_size", training_args.per_device_train_batch_size)
        mlflow.log_param("eval_batch_size", training_args.per_device_eval_batch_size)
        mlflow.log_param("gradient_accumulation_steps", training_args.gradient_accumulation_steps)
        mlflow.log_param("num_train_epochs", training_args.num_train_epochs)
        mlflow.log_param("weight_decay", training_args.weight_decay)
        mlflow.log_param("warmup_ratio", training_args.warmup_ratio)
        mlflow.log_param("fp16", training_args.fp16)

        # -------- Training --------
        trainer.train()

        # -------- Evaluation --------
        eval_metrics = trainer.evaluate()
        for key, value in eval_metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(key, value)

        # -------- Save model --------
        trainer.save_model(final_model_dir)
        tokenizer.save_pretrained(final_model_dir)

        # -------- Log artifacts --------
        mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
        )

        mlflow.log_artifacts(final_model_dir, artifact_path="hf_model")


if __name__ == "__main__":
    """
    Expected to be called with already prepared objects, e.g. from a pipeline:
        - tokenized_ds: DatasetDict
        - mlb: MultiLabelBinarizer
        - label_names: list[str]
        - metadata: dict (must include 'num_labels')
    """
    raise RuntimeError(
        "train_model(...) must be called from the training pipeline with prepared datasets."
    )
