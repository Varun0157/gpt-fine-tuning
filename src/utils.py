from torch.utils.data import Dataset, DataLoader

import pandas as pd


class SummarizationDataset(Dataset):
    def __init__(self, tokenizer, file_path, max_len=512):
        self.data = pd.read_csv(file_path, nrows=10)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        article = self.data.loc[idx, "article"]
        summary = self.data.loc[idx, "highlights"]

        # todo: check what the output of self.tokenizer is
        # todo: why do we only care about input_ids

        inputs = self.tokenizer(
            article,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        summary = self.tokenizer(
            summary,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": summary["input_ids"].squeeze(0),
        }


def get_dataloader(tokenizer, file_path, batch_size=8, max_length=512):
    dataset = SummarizationDataset(
        tokenizer=tokenizer, file_path=file_path, max_len=max_length
    )
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    return dataloader
