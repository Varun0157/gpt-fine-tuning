import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM


# todo: CausalLM uses CrossEntropyLoss, which ignores the index of -100 by default
IGNORE_INDEX = -100


class SoftPromptModel(nn.Module):
    def __init__(self, model_path, num_prompts=5, embedding_dim=768, device="cpu"):
        super(SoftPromptModel, self).__init__()
        self.device = device

        self.num_prompts = num_prompts
        self.embedding_dim = embedding_dim

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.gpt2 = AutoModelForCausalLM.from_pretrained(model_path)
        for param in self.gpt2.parameters():
            param.requires_grad = False

        self.soft_prompt_embeddings = nn.Embedding(num_prompts, embedding_dim)

        init_prompt = self.tokenizer.encode("[SUMMARIZE]", add_special_tokens=False)
        for i in range(min(num_prompts, len(init_prompt))):
            self.soft_prompt_embeddings.weight.data[i] = self.gpt2.transformer.wte(
                torch.tensor(init_prompt[i])
            )

    def forward(self, input_ids, attention_mask=None, labels=None):
        input_embeddings = self.gpt2.transformer.wte(input_ids)

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
        outputs = self.gpt2(
            inputs_embeds=extended_embeddings,
            attention_mask=extended_attention_mask,
            labels=extended_labels,
        )

        return outputs

    def _get_closest_tokens(self, embeddings):
        # Get the full vocabulary embeddings
        vocab_embeddings = (
            self.gpt2.transformer.wte.weight
        )  # (vocab_size, embedding_dim)

        # Reshape input embeddings to (batch_size * seq_len, embedding_dim)
        batch_size, seq_len, emb_dim = embeddings.shape
        flat_embeddings = embeddings.reshape(-1, emb_dim)

        # Normalize embeddings for cosine similarity
        vocab_norm = vocab_embeddings / vocab_embeddings.norm(dim=1, keepdim=True)
        flat_embeddings_norm = flat_embeddings / flat_embeddings.norm(
            dim=1, keepdim=True
        )

        # Compute cosine similarity
        similarity = torch.matmul(
            flat_embeddings_norm, vocab_norm.t()
        )  # (batch_size * seq_len, vocab_size)

        # Get the most similar tokens
        closest_tokens = similarity.argmax(dim=1)

        # Reshape back to (batch_size, seq_len)
        return closest_tokens.view(batch_size, seq_len)

    def generate(self, input_ids, attention_mask=None, **kwargs):
        batch_size = input_ids.size(0)

        # Get soft prompt embeddings
        soft_prompts = self.soft_prompt_embeddings.weight.unsqueeze(0).expand(
            batch_size, -1, -1
        )

        # Convert soft prompt embeddings to token IDs
        soft_prompt_ids = self._get_closest_tokens(soft_prompts)

        # Concatenate with input IDs
        extended_input_ids = torch.cat([soft_prompt_ids, input_ids], dim=1)

        if attention_mask is not None:
            extended_attention_mask = torch.cat(
                [
                    torch.ones(batch_size, self.num_prompts).to(input_ids.device),
                    attention_mask,
                ],
                dim=1,
            )
        else:
            extended_attention_mask = None

        outputs = self.gpt2.generate(
            extended_input_ids, attention_mask=extended_attention_mask, **kwargs
        )

        outputs = outputs[:, self.num_prompts :]

        return outputs
