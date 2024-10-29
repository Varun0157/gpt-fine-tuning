import torch.nn as nn
from peft import LoraConfig, get_peft_model  # type: ignore

from src.utils import get_frozen_model


class LoraTuning(nn.Module):
    def __init__(self, model_path, device):
        super(LoraTuning, self).__init__()
        self.device = device
        self.gpt2_neo = get_frozen_model(model_path, self.device)

        self.lora_config = LoraConfig(
            r=8,  # rank of the low-rank approximation
            lora_dropout=0.05,  # dropout for regularization
        )

        self.lora_model = get_peft_model(
            model=self.gpt2_neo,
            peft_config=self.lora_config,
        )

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.lora_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
