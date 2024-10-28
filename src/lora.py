import torch.nn as nn
from transformers import GPTNeoForCausalLM
from peft import LoraConfig, get_peft_model  # type: ignore


class LoraTuning(nn.Module):
    def __init__(self, model_path, device):
        super(LoraTuning, self).__init__()
        self.device = device
        self.gpt2_neo = GPTNeoForCausalLM.from_pretrained(model_path)
        for param in self.gpt2_neo.parameters():
            param.requires_grad = False
        self.gpt2_neo.to(self.device)

        self.lora_config = LoraConfig(
            r=8,
            lora_dropout=0.05,  # dropout for regularization
        )

        self.model = get_peft_model(
            model=self.gpt2_neo,
            peft_config=self.lora_config,
        )

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
