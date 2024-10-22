import torch
from soft_prompts import SoftPromptModel
from models import train_and_validate, test
from utils import get_dataloader
from transformers import AutoTokenizer

def main():
    # Set basic parameters
    model_dir = "./model"
    train_data = "./data/cnn_dailymail/train.csv"
    valid_data = "./data/cnn_dailymail/validation.csv"

    DESIRED_BATCH_SIZE = 64
    MAX_BATCHES_IN_MEM = 1
    assert DESIRED_BATCH_SIZE % MAX_BATCHES_IN_MEM == 0
    accumulation_steps = DESIRED_BATCH_SIZE // MAX_BATCHES_IN_MEM

    num_prompts = 5
    epochs = 10
    lr = 5e-3
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Initialize dataloader
    train_dataloader = get_dataloader(
        tokenizer, train_data, batch_size=MAX_BATCHES_IN_MEM
    )
    valid_dataloader = get_dataloader(
        tokenizer, valid_data, batch_size=MAX_BATCHES_IN_MEM
    )

    # Initialize soft prompt model
    model = SoftPromptModel(
        model_dir, num_prompts=num_prompts, embedding_dim=768, device=device
    )
    model = model.to(device)

    # Train the model
    train_and_validate(
        model,
        train_dataloader,
        valid_dataloader,
        epochs=epochs,
        lr=lr,
        accumulation_steps=accumulation_steps,
    )


if __name__ == "__main__":
    main()
