import torch.nn as nn

from src.utils import get_frozen_model


class GeneralTuning(nn.Module):
    def __init__(self, model_name, device):
        super(GeneralTuning, self).__init__()
        self.device = device
        self.gpt_neo = get_frozen_model(model_name, self.device)

    def forward(self, input_ids, attention_mask):
        raise NotImplementedError("method should be implemented in the child class")
