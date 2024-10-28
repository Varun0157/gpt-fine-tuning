import torch
import torch.nn as nn
from transformers import GPTNeoForCausalLM


class SoftPrompts(nn.Module):
    def __init__(
        self, num_soft_prompts, device, tokenizer, model_path, embedding_dim=768
    ):
        super(SoftPrompts, self).__init__()
        self.num_soft_prompts = num_soft_prompts
        self.device = device
        self.embedding_dim = embedding_dim

        self.soft_prompts = nn.Embedding(self.num_soft_prompts, self.embedding_dim)

        # initialise with [SUMMARIZE]
        init_embeddings = tokenizer.encode("[SUMMARIZE]", return_tensors="pt").to(
            device
        )
        self.soft_prompts.weight.data.copy_(init_embeddings.squeeze(0))

        # Load the pre-trained GPT-Neo model
        self.gpt2_neo = GPTNeoForCausalLM.from_pretrained(model_path)
        self.gpt2_neo.to(self.device)

        # Freeze all layers
        for param in self.gpt2_neo.parameters():
            param.requires_grad = False

        # print the number of trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"number of trainable parameters: {trainable_params}")

    def forward(self, input_ids):
        soft_prompt_ids = torch.arange(self.num_soft_prompts, device=self.device)
        soft_prompt_embeddings = self.soft_prompts(soft_prompt_ids).unsqueeze(0)
        assert soft_prompt_embeddings.shape == (
            1,
            self.num_soft_prompts,
            self.embedding_dim,
        )

        input_embeddings = self.gpt2_neo.transformer.wte(input_ids)
        input_embeddings = input_embeddings.to(self.device)

        embeddings = torch.cat([soft_prompt_embeddings, input_embeddings], dim=1)
        outputs = self.gpt2_neo(inputs_embeds=embeddings)

        return outputs.logits
