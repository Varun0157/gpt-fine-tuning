# general
Hey, thank you for pointing out this error. I made the mistake of assuming it’s a encoder-only model. This will lead to another error as if you were to run this code:

from transformers import GPT2Model, GPT2Tokenizer

model_name = 'gpt2'
model = GPT2Model.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

print(model)


You would see that for GPT-2 they don’t give a separate layer for the final linear layer that gives the logits (LM head):
GPT2Model(
  (wte): Embedding(50257, 768)
  (wpe): Embedding(1024, 768)
  (drop): Dropout(p=0.1, inplace=False)
  (h): ModuleList(
    (0-11): 12 x GPT2Block(
      (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (attn): GPT2SdpaAttention(
        (c_attn): Conv1D()
        (c_proj): Conv1D()
        (attn_dropout): Dropout(p=0.1, inplace=False)
        (resid_dropout): Dropout(p=0.1, inplace=False)
      )
      (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
      (mlp): GPT2MLP(
        (c_fc): Conv1D()
        (c_proj): Conv1D()
        (act): NewGELUActivation()
        (dropout): Dropout(p=0.1, inplace=False)
      )
    )
  )
  (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
)

Please change the mode to GPT-Neo 125M (https://huggingface.co/EleutherAI/gpt-neo-125m). If you were to run this code,

from transformers import GPTNeoForCausalLM, GPT2Tokenizer

model = GPTNeoForCausalLM.from_pretrained("EleutherAI/gpt-neo-125m")
tokenizer = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-125m")

print(model)

you would see that this mode has the final linear layer:

GPTNeoForCausalLM(
  (transformer): GPTNeoModel(
    (wte): Embedding(50257, 768)
    (wpe): Embedding(2048, 768)
    (drop): Dropout(p=0.0, inplace=False)
    (h): ModuleList(
      (0-11): 12 x GPTNeoBlock(
        (ln_1): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (attn): GPTNeoAttention(
          (attention): GPTNeoSelfAttention(
            (attn_dropout): Dropout(p=0.0, inplace=False)
            (resid_dropout): Dropout(p=0.0, inplace=False)
            (k_proj): Linear(in_features=768, out_features=768, bias=False)
            (v_proj): Linear(in_features=768, out_features=768, bias=False)
            (q_proj): Linear(in_features=768, out_features=768, bias=False)
            (out_proj): Linear(in_features=768, out_features=768, bias=True)
          )
        )
        (ln_2): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
        (mlp): GPTNeoMLP(
          (c_fc): Linear(in_features=768, out_features=3072, bias=True)
          (c_proj): Linear(in_features=3072, out_features=768, bias=True)
          (act): NewGELUActivation()
          (dropout): Dropout(p=0.0, inplace=False)
        )
      )
    )
    (ln_f): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
  )
  (lm_head): Linear(in_features=768, out_features=50257, bias=False)
)

Please try fine-tuning on this ‘lm_head’ alone. The assignment doc would be updated soon to reflect this.

_
Any decoder-only model works. In case don’t have access to lot of compute, try 125M models

___
# soft prompting
![Medium](./Harnessing%20the%20Power%20of%20Soft%20Prompts_%20A%20Hands-On%20Guide%20to%20Fine-Tuning%20for%20Text%20Summarization%20_%20by%20ANSHUL%20SHIVHARE%20_%20Medium.pdf)

___
# lora

To implement LoRA on GPT-2 small using PEFT libraries for parameter-efficient tuning, here’s the full process, including setting up the dataset class, dataloader, and LoRA fine-tuning:

### Step-by-Step Implementation

1. **Install Required Libraries**:
   Make sure you have the following libraries installed:

   ```bash
   pip install transformers peft datasets torch
   ```

2. **Get the Model and Tokenizer**:
   Using the function provided, we’ll load the tokenizer and model.

   ```python
   from transformers import AutoTokenizer, AutoModelForCausalLM

   def get_model_and_tokenizer(model_path):
       tokenizer = AutoTokenizer.from_pretrained(model_path)
       model = AutoModelForCausalLM.from_pretrained(model_path)
       return tokenizer, model

   tokenizer, model = get_model_and_tokenizer('./model')
   ```

3. **LoRA Implementation**:
   You can use the `peft` library to apply LoRA by adding low-rank matrices to the GPT-2 model. We'll freeze all parameters except for the LoRA layers.

   ```python
   from peft import LoraConfig, get_peft_model
   from transformers import AdamW

   # Define LoRA configuration
   lora_config = LoraConfig(
       r=8,  # Low-rank dimension (hyperparameter to tune)
       lora_alpha=16,  # Scaling factor
       lora_dropout=0.1,
       task_type="CAUSAL_LM"  # Task type is causal language modeling
   )

   # Add LoRA layers to GPT-2 and freeze other parameters
   model = get_peft_model(model, lora_config)

   # Freeze the rest of the model
   for param in model.base_model.parameters():
       param.requires_grad = False
   ```

4. **Dataset and DataLoader Setup**:
   Create a dataset class to handle the CNN/DailyMail dataset and store data in a DataLoader for easy batch processing.

   ```python
   import torch
   from torch.utils.data import Dataset, DataLoader
   import pandas as pd

   class CNNDailyMailDataset(Dataset):
       def __init__(self, data_file, tokenizer, max_length=512):
           self.data = pd.read_csv(data_file)
           self.tokenizer = tokenizer
           self.max_length = max_length

       def __len__(self):
           return len(self.data)

       def __getitem__(self, idx):
           article = self.data.iloc[idx]['article']
           summary = self.data.iloc[idx]['highlights']

           inputs = self.tokenizer(
               article, 
               max_length=self.max_length, 
               padding='max_length', 
               truncation=True, 
               return_tensors='pt'
           )

           labels = self.tokenizer(
               summary, 
               max_length=self.max_length, 
               padding='max_length', 
               truncation=True, 
               return_tensors='pt'
           ).input_ids

           labels[labels == self.tokenizer.pad_token_id] = -100  # Ignore padding in loss

           return {
               'input_ids': inputs.input_ids.squeeze(),
               'attention_mask': inputs.attention_mask.squeeze(),
               'labels': labels.squeeze()
           }

   # Prepare the dataset and dataloader
   def get_dataloaders(tokenizer, batch_size=8):
       train_dataset = CNNDailyMailDataset('./data/cnn_dailymail/train.csv', tokenizer)
       val_dataset = CNNDailyMailDataset('./data/cnn_dailymail/valid.csv', tokenizer)

       train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
       val_loader = DataLoader(val_dataset, batch_size=batch_size)

       return train_loader, val_loader

   train_loader, val_loader = get_dataloaders(tokenizer)
   ```

5. **Training Loop**:
   Now, we’ll define the training loop where only the LoRA layers are updated, while the rest of the GPT-2 model remains frozen.

   ```python
   from transformers import get_scheduler

   def train(model, train_loader, val_loader, tokenizer, num_epochs=3, learning_rate=5e-5):
       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       model.to(device)

       # Optimizer and learning rate scheduler
       optimizer = AdamW(model.parameters(), lr=learning_rate)
       num_training_steps = num_epochs * len(train_loader)
       lr_scheduler = get_scheduler(
           name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
       )

       # Training loop
       model.train()
       for epoch in range(num_epochs):
           for batch in train_loader:
               batch = {k: v.to(device) for k, v in batch.items()}

               outputs = model(**batch)
               loss = outputs.loss

               loss.backward()
               optimizer.step()
               lr_scheduler.step()
               optimizer.zero_grad()

           print(f"Epoch {epoch + 1}/{num_epochs} completed.")
       
       print("Training finished.")

   # Start training
   train(model, train_loader, val_loader, tokenizer)
   ```

### Summary of Components:
- **LoRA Configuration**: Adds low-rank adaptation layers to the GPT-2 model.
- **Dataset and DataLoader**: Handles the CNN/DailyMail dataset for summarization.
- **Training Loop**: Trains the model while keeping the base GPT-2 layers frozen and updating only the low-rank LoRA layers.

This setup enables you to efficiently fine-tune the GPT-2 model for summarization tasks using LoRA, keeping parameter updates minimal. Adjust the LoRA rank (`r`) to control the level of compression during tuning.
___
# traditional
To implement fine-tuning where only the `lm_head` layer of GPT-Neo 125M is trainable, while the rest of the model remains frozen, and to handle the CNN/DailyMail dataset for summarization, you can create a custom dataset class and a DataLoader.

Here’s an outline of the steps involved:

1. **Dataset Class**: Load the CSV files, tokenize the input (articles) and target (highlights). Also, you'll pad the sequences and create attention masks.
2. **Attention Masks**: Yes, you’ll need to generate attention masks so the model can properly handle variable-length sequences.
3. **Label Padding for Cross-Entropy**: For language modeling tasks, tokens that should not contribute to the loss (e.g., padding tokens) should be set to `-100` so that they are ignored during the loss computation.

Here’s the code that achieves this:

### Dataset Class and DataLoader

```python
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
import pandas as pd

class CNNDailyMailDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        self.data = pd.read_csv(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        article = str(self.data.loc[idx, 'article'])
        summary = str(self.data.loc[idx, 'highlights'])

        # Tokenize the article and summary
        inputs = self.tokenizer(
            article, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True,
            return_tensors="pt"
        )
        
        labels = self.tokenizer(
            summary, 
            max_length=self.max_length, 
            padding='max_length', 
            truncation=True,
            return_tensors="pt"
        )['input_ids']

        # Replace padding token id's in labels with -100 to ignore them during loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            'input_ids': inputs['input_ids'].squeeze(0),
            'attention_mask': inputs['attention_mask'].squeeze(0),
            'labels': labels.squeeze(0)
        }

def create_dataloader(file_path, tokenizer, batch_size=8):
    dataset = CNNDailyMailDataset(file_path, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
tokenizer.pad_token = tokenizer.eos_token  # Set the pad token to eos_token

# Example usage for train loader
train_loader = create_dataloader('./data/cnn_dailymail/train.csv', tokenizer, batch_size=8)
```

### Model Fine-Tuning (Freezing & Unfreezing)

Now, let's initialize the GPT-Neo model, freeze the necessary layers, and define the forward pass that returns the logits.

```python
import torch.nn as nn

class GPTNeoForSummarization(nn.Module):
    def __init__(self):
        super(GPTNeoForSummarization, self).__init__()
        # Load the pre-trained GPT-Neo model
        self.model = GPTNeoForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M')

        # Freeze all layers except the `lm_head`
        for param in self.model.parameters():
            param.requires_grad = False

        # Only allow the `lm_head` (the output layer) to be fine-tuned
        for param in self.model.lm_head.parameters():
            param.requires_grad = True

    def forward(self, input_ids, attention_mask, labels=None):
        # Forward pass through the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        # `logits` contains the predicted output
        return outputs.logits

# Instantiate the model
model = GPTNeoForSummarization()
```

### Loss and Training
Since you're using cross-entropy loss, and the `labels` have padding tokens replaced by `-100`, you don't need to manually adjust the indices during training.

```python
from torch.optim import AdamW

# Define optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Training loop (just a basic example)
for epoch in range(3):  # Assuming 3 epochs
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        # Forward pass
        logits = model(input_ids, attention_mask, labels=labels)

        # Compute loss
        loss = nn.CrossEntropyLoss()(logits.view(-1, logits.size(-1)), labels.view(-1))
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss.item()}")
```

### Key Points:
- **Dataloader**: Each batch consists of `input_ids`, `attention_mask`, and `labels`.
- **Freezing**: All layers are frozen except the `lm_head`, which is fine-tuned.
- **Padding**: Padding tokens in `labels` are replaced with `-100` for ignoring during loss computation.

Let me know if you need further adjustments!
