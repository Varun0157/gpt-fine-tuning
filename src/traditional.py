import torch.nn as nn
from transformers import GPTNeoForCausalLM


class TraditionalTuning(nn.Module):
    def __init__(self, device, model_path):
        super(TraditionalTuning, self).__init__()
        self.device = device

        # Load the pre-trained GPT-Neo model
        self.gpt2_neo = GPTNeoForCausalLM.from_pretrained(model_path)
        self.gpt2_neo.to(self.device)

        # Freeze all layers except the `lm_head`
        for param in self.gpt2_neo.parameters():
            param.requires_grad = False

        # Only allow the `lm_head` (the output layer) to be fine-tuned
        for param in self.gpt2_neo.lm_head.parameters():
            param.requires_grad = True

        print(
            "num_params after freezing: ",
            sum(p.numel() for p in self.gpt2_neo.parameters() if p.requires_grad),
        )

    def forward(self, input_ids, attention_mask, labels=None):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        # Forward pass through the model
        outputs = self.gpt2_neo(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        # `logits` contains the predicted output
        return outputs.logits
