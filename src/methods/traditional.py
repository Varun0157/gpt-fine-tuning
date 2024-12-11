import torch.nn as nn

from src.utils import get_frozen_model


class TraditionalTuning(nn.Module):
    def __init__(self, device, model_path):
        super(TraditionalTuning, self).__init__()
        self.device = device

        # load the pre-trained GPT-Neo model
        self.gpt2_neo = get_frozen_model(model_path, self.device)

        # Only allow the `lm_head` (the output layer) to be fine-tuned
        for param in self.gpt2_neo.lm_head.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, labels=None):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Forward pass through the model
        outputs = self.gpt2_neo(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        # `logits` contains the predicted output
        return outputs.logits
