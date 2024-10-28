from enum import Enum
from string import punctuation as PUNCTUATION

from torch.utils.data import Dataset, DataLoader
import pandas as pd

from src.utils import IGNORE_INDEX


class DatasetType(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"


def _clean_text(text):
    sentences = text.split("\n")
    if len(sentences) > 1:
        return " . ".join([_clean_text(sentence) for sentence in sentences]).strip()

    for ch in PUNCTUATION:  # removed - for instances like well-known
        text = text.replace(ch, " ")

    return text.strip()


class CNNDailyMailDataset(Dataset):
    def __init__(self, file_path, tokenizer, DatasetType, max_length=256):
        if DatasetType == DatasetType.TRAIN:
            num_rows = 70
        elif DatasetType == DatasetType.VALIDATION:
            num_rows = 20
        else:
            num_rows = 10

        self.data = pd.read_csv(file_path, nrows=num_rows)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        article = _clean_text(str(self.data.loc[idx, "article"]))
        summary = _clean_text(str(self.data.loc[idx, "highlights"]))

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
        labels[labels == self.tokenizer.pad_token_id] = IGNORE_INDEX

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels.squeeze(0),
        }


def create_dataloader(file_path, tokenizer, DatasetType, batch_size):
    dataset = CNNDailyMailDataset(file_path, tokenizer, DatasetType)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)
