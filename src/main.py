import torch

from transformers import GPT2Tokenizer

from src.data import create_dataloader
from src.traditional import GPTNeoForSummarization
from src.utils import train_and_validate

MAX_BATCH_IN_MEM = 1


def main():
    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("./model")
    tokenizer.pad_token = tokenizer.eos_token  # Set the pad token to eos_token

    # Example usage for train loader
    train_loader = create_dataloader(
        "./data/cnn_dailymail/train.csv", tokenizer, batch_size=MAX_BATCH_IN_MEM
    )
    valid_dataloader = create_dataloader(
        "./data/cnn_dailymail/valid.csv", tokenizer, batch_size=MAX_BATCH_IN_MEM
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GPTNeoForSummarization(device)
    model.to(device)

    train_and_validate(model, train_loader, valid_dataloader, num_epochs=10, lr=5e-3)


if __name__ == "__main__":
    main()
