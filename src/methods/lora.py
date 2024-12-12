from peft import LoraConfig, get_peft_model  # type: ignore

from src.methods.general import GeneralTuning


class LoraTuning(GeneralTuning):
    def __init__(self, model_name, device):
        super().__init__(model_name, device)

        self.lora_config = LoraConfig(
            r=8,  # rank of the low-rank approximation
            lora_dropout=0.05,  # dropout for regularization
        )

        self.model = get_peft_model(
            model=self.gpt_neo,
            peft_config=self.lora_config,
        )

    def forward(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits
