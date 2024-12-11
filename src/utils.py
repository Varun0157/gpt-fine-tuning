import os
import time
from enum import Enum
from typing import Tuple
import logging

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import GPTNeoForCausalLM, GPT2Tokenizer
from evaluate import load

IGNORE_INDEX = -100


def _train(model, dataloader, criterion, optimizer, accumulation_steps: int = 1):
    model.train()
    optimizer.zero_grad()

    total_loss = 0
    for step, batch in enumerate(dataloader):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)

        # Forward pass
        logits = model(input_ids, attention_mask)
        logits = logits.view(-1, logits.shape[-1])

        labels = labels.view(-1)

        loss = criterion(logits, labels)
        loss.backward()
        total_loss += loss.item()

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(dataloader):
            optimizer.step()
            optimizer.zero_grad()

    NUM_SAMPLES = len(dataloader.dataset)
    return total_loss / NUM_SAMPLES


def _validate(model, dataloader, criterion):
    model.eval()

    total_loss = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)

        with torch.no_grad():
            logits = model(input_ids, attention_mask)
            logits = logits.view(-1, logits.shape[-1])

            labels = labels.view(-1)

            loss = criterion(logits, labels)
            total_loss += loss.item()

    NUM_SAMPLES = len(dataloader.dataset)
    return total_loss / NUM_SAMPLES


def train_and_validate(
    model,
    train_dataloader,
    valid_dataloader,
    best_model_path,
    num_epochs=10,
    lr=7.5e-4,
    accumulation_steps: int = 1,
    patience: int = 2,
):
    model.train()

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # Initialize AdamW optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    best_valid_loss = float("inf")
    num_val_inc_epochs = 0
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = _train(
            model=model,
            dataloader=train_dataloader,
            criterion=criterion,
            optimizer=optimizer,
            accumulation_steps=accumulation_steps,
        )
        valid_loss = _validate(model, valid_dataloader, criterion)
        end_time = time.time()

        logging.info(
            f"epoch {epoch + 1} -> train loss: {train_loss}, valid loss: {valid_loss} in {end_time - start_time} seconds"
        )

        if valid_loss > best_valid_loss:
            num_val_inc_epochs += 1
            if num_val_inc_epochs < patience:
                continue

            print("early stopping")
            break

        num_val_inc_epochs = 0
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), best_model_path)
        logging.info("\tmodel saved")


def test(model, tokenizer, test_dataloader):
    model.eval()

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    all_predictions, all_references = [], []

    total_loss = 0
    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)

        labels = batch["labels"].to(model.device)  # (b, s)
        batch_labels = labels.view(-1)  # (b * s)

        with torch.no_grad():
            logits = model(
                input_ids=input_ids, attention_mask=attention_mask
            )  # (b, s, v)
            batch_logits = logits.view(-1, logits.shape[-1])  # (b * s, v)

            loss = criterion(batch_logits, batch_labels)
            total_loss += loss.item()

            # todo: compute allows you to use batches, so maybe make this batched instead
            # todo:     of iterating through the batches
            # todo: https://huggingface.co/docs/evaluate/v0.4.0/en/a_quick_tour#compute
            for i in range(logits.size(0)):  # batch size
                predictions = logits[i].argmax(dim=-1).tolist()
                references = labels[i].tolist()

                # Remove padding and ignore indices from references
                references = [token for token in references if token != IGNORE_INDEX]

                all_predictions.append(
                    tokenizer.decode(predictions, skip_special_tokens=True)
                )
                all_references.append(
                    tokenizer.decode(references, skip_special_tokens=True)
                )

    NUM_SAMPLES = len(test_dataloader.dataset)
    logging.info(f"test loss: {total_loss / NUM_SAMPLES}")

    rouge = load("rouge")
    results = rouge.compute(
        predictions=all_predictions,
        references=all_references,
    )

    print()

    if results is None:
        logging.warning("unable to calculate rouge score")
        return
    logging.info("rouge scores ->")
    for key, val in results.items():
        logging.info(f"\t{key}: {val}")


def get_frozen_model(model_path, device):
    model = GPTNeoForCausalLM.from_pretrained(model_path)
    for param in model.parameters():
        param.requires_grad = False

    return model.to(device)


def get_tokenizer(tokenizer_path):
    tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.eos_token  # Set the pad token to eos_token

    return tokenizer


class FineTuningType(Enum):
    TRADITIONAL = "traditional"
    SOFT_PROMPTS = "soft_prompts"
    LORA = "lora"


def get_tuned_model_path(type: FineTuningType) -> str:
    return os.path.join("res", f"{type.value}.pth")


def get_base_paths() -> Tuple[str, str]:
    model_path = os.path.join("model", "model")
    tokenizer_path = os.path.join("model", "tokenizer")

    return model_path, tokenizer_path


def get_logging_format() -> str:
    return "%(asctime)s - %(levelname)s : %(message)s"
