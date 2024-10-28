import torch
from transformers import GPT2Tokenizer

from src.data import DatasetType, create_dataloader
from src.utils import get_tokenizer, train_and_validate, test, FineTuningType
from src.traditional import TraditionalTuning
from src.soft_prompting import SoftPromptTuning
from src.lora import LoraTuning

MAX_BATCH_IN_MEM = 2
DESIRED_BATCH_SIZE = 64

assert DESIRED_BATCH_SIZE % MAX_BATCH_IN_MEM == 0
ACCUMULATION_STEPS = DESIRED_BATCH_SIZE // MAX_BATCH_IN_MEM


def fine_tune(tuning_type: FineTuningType, lr: float = 5e-4, num_epochs: int = 10):
    print(f"learning rate: {lr}")
    print(f"batch size: {MAX_BATCH_IN_MEM} in mem, {DESIRED_BATCH_SIZE} for optimizer")
    print()

    MODEL_PATH = "./model/model"
    TOKEN_PATH = "./model/tokenizer"

    # Initialize tokenizer
    tokenizer = get_tokenizer(TOKEN_PATH)

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
        model = SoftPromptTuning(
            device=device,
            tokenizer=tokenizer,
            model_path=MODEL_PATH,
        )
        BEST_MODEL_PATH = "soft_prompts.pth"
    elif tuning_type == FineTuningType.TRADITIONAL:
        model = TraditionalTuning(device=device, model_path=MODEL_PATH)
        BEST_MODEL_PATH = "traditional.pth"
    elif tuning_type == FineTuningType.LORA:
        model = LoraTuning(model_path=MODEL_PATH, device=device)
        BEST_MODEL_PATH = "lora.pth"
    else:
        print("no such fine tuning method")
        return

    model.to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"number of trainable parameters: {trainable_params}")

    train_and_validate(
        model,
        train_loader,
        valid_dataloader,
        BEST_MODEL_PATH,
        num_epochs=num_epochs,
        lr=lr,
        accumulation_steps=ACCUMULATION_STEPS,
    )

    max_gpu_mb_alloc = torch.cuda.max_memory_allocated() / 1024**2
    print(f"Max GPU memory allocated: {max_gpu_mb_alloc:.2f} MB")
    # torch.cuda.reset_peak_memory_stats() also exists, may be useful


if __name__ == "__main__":
    fine_tune(FineTuningType.LORA)
