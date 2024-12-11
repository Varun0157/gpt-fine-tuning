import torch.nn as nn

from src.methods.general import GeneralTuning
from src.utils import get_frozen_model


class TraditionalTuning(GeneralTuning):
    def __init__(self, device, model_path):
        super().__init__(model_path, device)
        # Only allow the `lm_head` (the output layer) to be fine-tuned
        for param in self.gpt_neo.lm_head.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, labels=None):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Forward pass through the model
        outputs = self.gpt_neo(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        # `logits` contains the predicted output
        return outputs.logits
