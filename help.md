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