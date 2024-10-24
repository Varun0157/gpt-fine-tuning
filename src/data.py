from torch.utils.data import Dataset, DataLoader
import pandas as pd


class CNNDailyMailDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=384):
        self.data = pd.read_csv(file_path, nrows=10)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        article = str(self.data.loc[idx, "article"])
        summary = str(self.data.loc[idx, "highlights"])

        # Tokenize the article and summary
        inputs = self.tokenizer(
            article,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = self.tokenizer(
            summary,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )["input_ids"]

        # Replace padding token id's in labels with -100 to ignore them during loss calculation
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


def create_dataloader(file_path, tokenizer, batch_size=8):
    dataset = CNNDailyMailDataset(file_path, tokenizer)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
