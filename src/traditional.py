import torch.nn as nn
from transformers import GPTNeoForCausalLM


class GPTNeoForSummarization(nn.Module):
    def __init__(self, device):
        super(GPTNeoForSummarization, self).__init__()
        self.device = device
        
        # Load the pre-trained GPT-Neo model
        self.model = GPTNeoForCausalLM.from_pretrained("./model")
        self.model.to(self.device)

        # Freeze all layers except the `lm_head`
        for param in self.model.parameters():
            param.requires_grad = False

        # Only allow the `lm_head` (the output layer) to be fine-tuned
        for param in self.model.lm_head.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, labels=None):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        # Forward pass through the model
        outputs = self.model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )

        # `logits` contains the predicted output
        return outputs.logits
