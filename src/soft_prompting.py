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

        # Load the pre-trained GPT-Neo model
        self.gpt2_neo = GPTNeoForCausalLM.from_pretrained(model_path)
        # Freeze all layers
        for param in self.gpt2_neo.parameters():
            param.requires_grad = False
        self.gpt2_neo.to(self.device)

        self.soft_prompts = nn.Embedding(self.num_soft_prompts, self.embedding_dim)

        # initialise with [SUMMARIZE]
        init_embeddings = self.gpt2_neo.transformer.wte(
            tokenizer.encode(
                "[SUMMARIZE THIS ARTICLE INTO HIGHLIGHTS]", return_tensors="pt"
            ).to(device)
        )
        if init_embeddings.shape[1] != self.num_soft_prompts:
            init_embeddings = init_embeddings[:, : self.num_soft_prompts, :]

        self.soft_prompts.weight.data.copy_(init_embeddings.squeeze(0))

        # print the number of trainable parameters
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"number of trainable parameters: {trainable_params}")

    def forward(self, input_ids, attention_mask):
        BATCH_SIZE, SEQ_LEN = input_ids.shape
        input_embeddings = self.gpt2_neo.transformer.wte(input_ids)
        input_embeddings = input_embeddings.to(self.device)
        # prepend to the attention mask to account for the soft prompt
        attention_mask = torch.cat(
            [
                torch.ones(BATCH_SIZE, self.num_soft_prompts, device=self.device),
                attention_mask,
            ],
            dim=1,
        )

        soft_prompt_ids = torch.arange(self.num_soft_prompts, device=self.device)
        soft_prompt_embeddings = self.soft_prompts(soft_prompt_ids)
        soft_prompt_embeddings = soft_prompt_embeddings.unsqueeze(0).expand(
            BATCH_SIZE, -1, -1
        )

        embeddings = torch.cat([soft_prompt_embeddings, input_embeddings], dim=1)
        # truncate the embeddings to sequence length
        embeddings = embeddings[:, :SEQ_LEN, :]
        attention_mask = attention_mask[:, :SEQ_LEN]

        outputs = self.gpt2_neo(inputs_embeds=embeddings, attention_mask=attention_mask)

        return outputs.logits
