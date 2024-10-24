import time

import torch
import torch.nn as nn
from torch.optim import AdamW


def _train(model, dataloader, criterion, optimizer):
    model.train()

    total_loss = 0
    for batch in dataloader:
        optimizer.zero_grad()

        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)

        # Forward pass
        logits = model(input_ids, attention_mask)
        logits = logits.view(-1, logits.shape[-1])

        labels = labels.view(-1)

        loss = criterion(logits, labels)
        total_loss += loss.item()

        loss.backward()
        optimizer.step()

    return total_loss


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

    return total_loss


def train_and_validate(
    model, train_dataloader, valid_dataloader, num_epochs=3, lr=5e-4
):
    model.train()

    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    # Initialize AdamW optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    best_model_path = "best_model.pth"
    best_valid_loss = float("inf")
    for epoch in range(num_epochs):
        start_time = time.time()
        train_loss = _train(model, train_dataloader, criterion, optimizer)
        valid_loss = _validate(model, valid_dataloader, criterion)
        end_time = time.time()

        print(
            f"epoch {epoch + 1} -> train loss: {train_loss}, valid loss: {valid_loss} in {end_time - start_time} seconds"
        )

        if valid_loss > best_valid_loss:
            continue

        best_valid_loss = valid_loss
        torch.save(model.state_dict(), best_model_path)
        print("\tbest model saved")


def _test(model, test_dataloader):
    pass
