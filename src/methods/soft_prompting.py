import torch
import torch.nn as nn

from src.utils import get_frozen_model


class SoftPromptTuning(nn.Module):
    def __init__(
        self,
        device,
        tokenizer,
        model_path,
        num_soft_prompts: int = 12,
        embedding_dim=768,
    ):
        super(SoftPromptTuning, self).__init__()
        self.num_soft_prompts = num_soft_prompts
        self.device = device
        self.embedding_dim = embedding_dim

        # Load the pre-trained GPT-Neo model
        self.gpt2_neo = get_frozen_model(model_path, self.device)

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

    def forward(self, input_ids, attention_mask):
        BATCH_SIZE, SEQ_LEN = input_ids.shape
        input_embeddings = self.gpt2_neo.transformer.wte(input_ids)
        input_embeddings = input_embeddings.to(self.device)

        soft_prompt_ids = torch.arange(self.num_soft_prompts, device=self.device)
        soft_prompt_embeddings = self.soft_prompts(soft_prompt_ids)
        soft_prompt_embeddings = soft_prompt_embeddings.unsqueeze(0).expand(
            BATCH_SIZE, -1, -1
        )

        embeddings = torch.cat([soft_prompt_embeddings, input_embeddings], dim=1)
        # prepending ones so the soft prompts are not ignored
        attention_mask = torch.cat(
            [
                torch.ones(BATCH_SIZE, self.num_soft_prompts, device=self.device),
                attention_mask,
            ],
            dim=1,
        )
        # truncate the embeddings and mask to sequence length
        embeddings = embeddings[:, :SEQ_LEN, :]
        attention_mask = attention_mask[:, :SEQ_LEN]

        outputs = self.gpt2_neo(inputs_embeds=embeddings, attention_mask=attention_mask)
        return outputs.logits
