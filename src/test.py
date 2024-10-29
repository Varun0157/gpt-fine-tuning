import torch

from src.data import DatasetType, create_dataloader
from src.utils import FineTuningType, get_tokenizer, test
from src.traditional import TraditionalTuning
from src.soft_prompting import SoftPromptTuning
from src.lora import LoraTuning


def test_tuned_model(
    checkpoint_path: str, batch_size: int, tuning_type: FineTuningType
):
    MODEL_PATH = "./model/model"
    TOKEN_PATH = "./model/tokenizer"
    tokenizer = get_tokenizer(TOKEN_PATH)

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
            model_path=MODEL_PATH,
        )
    elif tuning_type == FineTuningType.TRADITIONAL:
        model = TraditionalTuning(device=device, model_path=MODEL_PATH)
    elif tuning_type == FineTuningType.LORA:
        model = LoraTuning(model_path=MODEL_PATH, device=device)
    else:
        print("no such fine tuning method")
        return
    model.to(device)

    checkpoint = torch.load(checkpoint_path, weights_only=True)
    model.load_state_dict(checkpoint)

    test(model, tokenizer, test_dataloader)


if __name__ == "__main__":
    test_tuned_model("lora.pth", 2, FineTuningType.LORA)
