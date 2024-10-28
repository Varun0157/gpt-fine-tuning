import os
from enum import Enum

import torch
from transformers import GPT2Tokenizer

from src.data import DatasetType, create_dataloader
from src.utils import train_and_validate, test
from src.traditional import TraditionalTuning
from src.soft_prompting import SoftPrompts

MAX_BATCH_IN_MEM = 2
DESIRED_BATCH_SIZE = 64

assert DESIRED_BATCH_SIZE % MAX_BATCH_IN_MEM == 0
ACCUMULATION_STEPS = DESIRED_BATCH_SIZE / MAX_BATCH_IN_MEM


class FineTuningType(Enum):
    TRADITIONAL = "traditional"
    SOFT_PROMPTING = "soft_prompting"


def fine_tune(tuning_type: FineTuningType):
    MODEL_PATH = "./model/model"
    TOKEN_PATH = "./model/tokenizer"

    # Initialize tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(TOKEN_PATH)
    tokenizer.pad_token = tokenizer.eos_token  # Set the pad token to eos_token

    # Example usage for train loader
    train_loader = create_dataloader(
        "./data/cnn_dailymail/train.csv",
        tokenizer,
        DatasetType=DatasetType.TRAIN,
        batch_size=MAX_BATCH_IN_MEM,
    )
    valid_dataloader = create_dataloader(
        "./data/cnn_dailymail/validation.csv",
        tokenizer,
        DatasetType=DatasetType.VALIDATION,
        batch_size=MAX_BATCH_IN_MEM,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if tuning_type == FineTuningType.SOFT_PROMPTING:
        model = SoftPrompts(
            num_soft_prompts=5,
            device=device,
            tokenizer=tokenizer,
            model_path=MODEL_PATH,
            embedding_dim=768,
        )
        BEST_MODEL_PATH = "soft_prompts.pth"
    elif tuning_type == FineTuningType.TRADITIONAL:
        model = TraditionalTuning(device=device, model_path=MODEL_PATH)
        BEST_MODEL_PATH = "traditional.pth"
    else:
        print("no such fine tuning method")
        return

    model.to(device)

    train_and_validate(model, train_loader, valid_dataloader, BEST_MODEL_PATH)

    max_gpu_mb_alloc = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Max GPU memory allocated: {max_gpu_mb_alloc:.2f} MB")
    # torch.cuda.reset_peak_memory_stats() also exists, may be useful

    # todo: separate the train and test later
    test_dataloader = create_dataloader(
        "./data/cnn_dailymail/test.csv",
        tokenizer,
        DatasetType=DatasetType.TEST,
        batch_size=MAX_BATCH_IN_MEM,
    )
    checkpoint = torch.load(BEST_MODEL_PATH, weights_only=True)
    model.load_state_dict(checkpoint)

    test(model, tokenizer, test_dataloader)


if __name__ == "__main__":
    fine_tune(FineTuningType.SOFT_PROMPTING)
