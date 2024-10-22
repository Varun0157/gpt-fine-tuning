import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


# CausalLM uses CrossEntropyLoss, which ignores the index of -100 by default
IGNORE_INDEX = -100


class SoftPromptModel(nn.Module):
    def __init__(self, model_path, num_prompts=5, embedding_dim=768, device="cpu"):
        super(SoftPromptModel, self).__init__()
        self.device = device

        self.num_prompts = num_prompts
        self.embedding_dim = embedding_dim

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        for param in self.model.parameters():
            param.requires_grad = False

        self.soft_prompt_embeddings = nn.Embedding(num_prompts, embedding_dim)

        init_prompt = self.tokenizer.encode("[SUMMARIZE]", add_special_tokens=False)
        for i in range(min(num_prompts, len(init_prompt))):
            self.soft_prompt_embeddings.weight.data[i] = self.model.transformer.wte(
                torch.tensor(init_prompt[i])
            )

    def forward(self, input_ids, attention_mask=None, labels=None):
        input_embeddings = self.model.transformer.wte(input_ids)

        soft_prompts = self.soft_prompt_embeddings.weight.unsqueeze(0).expand(
            input_embeddings.size(0), -1, -1
        )
        extended_embeddings = torch.cat([soft_prompts, input_embeddings], dim=1)

        if attention_mask is not None:
            extended_attention_mask = torch.cat(
                [
                    torch.ones(input_embeddings.size(0), self.num_prompts).to(
                        input_embeddings.device
                    ),
                    attention_mask,
                ],
                dim=1,
            )
        else:
            extended_attention_mask = None

        if labels is not None:
            extended_labels = torch.cat(
                [
                    torch.full((labels.size(0), self.num_prompts), -100).to(
                        labels.device
                    ),
                    labels,
                ],
                dim=1,
            )
        else:
            extended_labels = None

        # Pass the extended embeddings to the model
        outputs = self.model(
            inputs_embeds=extended_embeddings,
            attention_mask=extended_attention_mask,
            labels=extended_labels,
        )

        return outputs
