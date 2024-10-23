import torch
from soft_prompts import SoftPromptModel
from utils import evaluate_model, train_and_validate, test
from data import get_dataloader
from transformers import AutoTokenizer

# Set basic parameters
model_dir = "./model"
train_data = "./data/cnn_dailymail/train.csv"
valid_data = "./data/cnn_dailymail/validation.csv"

DESIRED_BATCH_SIZE = 64
MAX_BATCHES_IN_MEM = 2
assert DESIRED_BATCH_SIZE % MAX_BATCHES_IN_MEM == 0
accumulation_steps = DESIRED_BATCH_SIZE // MAX_BATCHES_IN_MEM

num_prompts = 5
epochs = 3
lr = 5e-3
device = "cuda" if torch.cuda.is_available() else "cpu"


def train(model, tokenizer):
    # Initialize dataloader
    train_dataloader = get_dataloader(
        tokenizer, train_data, batch_size=MAX_BATCHES_IN_MEM
    )
    valid_dataloader = get_dataloader(
        tokenizer, valid_data, batch_size=MAX_BATCHES_IN_MEM
    )

    # Train the model
    train_and_validate(
        model,
        train_dataloader,
        valid_dataloader,
        epochs=epochs,
        lr=lr,
        accumulation_steps=accumulation_steps,
    )


def test(model, tokenizer):
    test_data = "./data/cnn_dailymail/test.csv"
    test_dataloader = get_dataloader(
        tokenizer, test_data, batch_size=MAX_BATCHES_IN_MEM
    )
    checkpoint = torch.load("best_model.pth", weights_only=True)
    model.load_state_dict(checkpoint)
    evaluate_model(model, test_dataloader, tokenizer, device)


def main():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Initialize soft prompt model
    model = SoftPromptModel(
        model_dir, num_prompts=num_prompts, embedding_dim=768, device=device
    )
    model = model.to(device)
    model.gpt2.generation_config.pad_token_id = tokenizer.pad_token_id

    print("beginning training")
    train(model, tokenizer)
    print("beginning testing")
    test(model, tokenizer)


if __name__ == "__main__":
    main()