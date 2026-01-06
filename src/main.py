import os
import torch

from bert.finetune_bert_model import train


def main():
    print("=" * 60)
    print("Multi-head RoBERTa fine-tuning")
    print("=" * 60)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train()


if __name__ == "__main__":
    main()
