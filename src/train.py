import torch
import logging

from src.data import DatasetType, create_dataloader
from src.utils import (
    get_base_paths,
    get_logging_format,
    get_tuned_model_path,
    get_tokenizer,
    train_and_validate,
    FineTuningType,
)
from src.methods.traditional import TraditionalTuning
from src.methods.soft_prompting import SoftPromptTuning
from src.methods.lora import LoraTuning

MAX_BATCH_IN_MEM = 2
DESIRED_BATCH_SIZE = 64

assert DESIRED_BATCH_SIZE % MAX_BATCH_IN_MEM == 0
ACCUMULATION_STEPS = DESIRED_BATCH_SIZE // MAX_BATCH_IN_MEM


def fine_tune(tuning_type: FineTuningType, lr: float = 5e-4, num_epochs: int = 10):
    logging.info(f"learning rate: {lr}")
    logging.info(
        f"batch size: {MAX_BATCH_IN_MEM} in mem, {DESIRED_BATCH_SIZE} for optimizer"
    )
    print()

    MODEL_PATH, TOKEN_PATH = get_base_paths()

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
    if tuning_type == FineTuningType.SOFT_PROMPTS:
        model = SoftPromptTuning(
            device=device,
            tokenizer=tokenizer,
            model_path=MODEL_PATH,
        )
    elif tuning_type == FineTuningType.TRADITIONAL:
        model = TraditionalTuning(device=device, model_path=MODEL_PATH)
    elif tuning_type == FineTuningType.LORA:
        model = LoraTuning(device=device, model_path=MODEL_PATH)
    else:
        logging.warning("no such fine tuning method")
        return
    BEST_MODEL_PATH = get_tuned_model_path(tuning_type)

    model.to(device)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f"number of trainable parameters: {trainable_params}")

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
    logging.info(f"Max GPU memory allocated: {max_gpu_mb_alloc:.2f} MB")
    # NOTE: torch.cuda.reset_peak_memory_stats() also exists, may be useful


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format=get_logging_format(),
    )

    import argparse

    parser = argparse.ArgumentParser(description="fine-tune gpt-neo")
    parser.add_argument(
        "--fine_tuning_type",
        type=str,
        required=True,
        choices=[t.value for t in FineTuningType],
        help="type of fine-tuning",
    )
    args = parser.parse_args()

    fine_tune(FineTuningType(args.fine_tuning_type))
