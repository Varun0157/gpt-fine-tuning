import time

import torch
from torch.optim import AdamW


def train(model, train_dataloader, optimizer, accumulation_steps=1):
    model.train()

    total_loss = 0
    optimizer.zero_grad()

    for step, batch in enumerate(train_dataloader):
        input_ids = batch["input_ids"].to(model.device)
        attention_mask = batch["attention_mask"].to(model.device)
        labels = batch["labels"].to(model.device)

        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss

        loss.backward()
        total_loss += loss.item()

        if (step + 1) % accumulation_steps == 0 or (step + 1) == len(train_dataloader):
            optimizer.step()
            optimizer.zero_grad()

    return total_loss


def validate(model, val_dataloader):
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()

    return total_loss


def train_and_validate(
    model, train_dataloader, valid_dataloader, epochs=10, lr=5e-4, accumulation_steps=1
):
    optimizer = AdamW(model.parameters(), lr=lr)

    best_model_path = "best_model.pth"
    best_val_loss = float("inf")

    for epoch in range(epochs):
        start_time = time.time()
        train_loss = train(
            model, train_dataloader, optimizer, accumulation_steps=accumulation_steps
        )
        val_loss = validate(model, valid_dataloader)
        time_taken = time.time() - start_time

        print(
            f"epoch: {epoch} -> train loss: {train_loss}, val loss: {val_loss} in {time_taken} seconds"
        )

        if val_loss > best_val_loss:
            continue

        best_val_loss = val_loss
        torch.save(model.state_dict(), best_model_path)
        print("\tmodel saved")


def test(model, test_dataloader):
    model.eval()

    total_loss = 0
    with torch.no_grad():
        for batch in test_dataloader:
            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)
            labels = batch["labels"].to(model.device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            total_loss += loss.item()

    return total_loss
