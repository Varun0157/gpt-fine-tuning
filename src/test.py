import logging
import torch

from src.data import DatasetType, create_dataloader
from src.utils import (
    FineTuningType,
    get_base_model,
    get_logging_format,
    get_tokenizer,
    get_tuned_model_path,
    test,
)
from src.methods.traditional import TraditionalTuning
from src.methods.soft_prompting import SoftPromptTuning
from src.methods.lora import LoraTuning


def test_tuned_model(tuning_type: FineTuningType, batch_size: int):
    MODEL_NAME = get_base_model()
    tokenizer = get_tokenizer(MODEL_NAME)

    test_dataloader = create_dataloader(
        "./data/cnn_dailymail/test.csv",
        tokenizer,
        DatasetType=DatasetType.TEST,
        batch_size=batch_size,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if tuning_type == FineTuningType.SOFT_PROMPTS:
        model = SoftPromptTuning(
            device=device,
            tokenizer=tokenizer,
            model_name=MODEL_NAME,
        )
    elif tuning_type == FineTuningType.TRADITIONAL:
        model = TraditionalTuning(device=device, model_name=MODEL_NAME)
    elif tuning_type == FineTuningType.LORA:
        model = LoraTuning(model_name=MODEL_NAME, device=device)

    model.to(device)

    checkpoint_path = get_tuned_model_path(tuning_type)
    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint)

    test(model, tokenizer, test_dataloader)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format=get_logging_format())

    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune a model")
    parser.add_argument(
        "--fine_tuning_type",
        type=str,
        required=True,
        choices=[t.value for t in FineTuningType],
        help="type of fine-tuning",
    )
    args = parser.parse_args()

    test_tuned_model(FineTuningType(args.fine_tuning_type), 2)
