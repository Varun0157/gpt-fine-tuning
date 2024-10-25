import time

import torch
import torch.nn as nn
from torch.optim import AdamW

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
    model,
    train_dataloader,
    valid_dataloader,
    num_epochs=3,
    lr=5e-4,
    accumulation_steps: int = 1,
):
    model.train()

    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX)

    # Initialize AdamW optimizer
    optimizer = AdamW(model.parameters(), lr=lr)

    best_model_path = "best_model.pth"
    best_valid_loss = float("inf")
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

        print(
            f"epoch {epoch + 1} -> train loss: {train_loss}, valid loss: {valid_loss} in {end_time - start_time} seconds"
        )

        if valid_loss > best_valid_loss:
            continue

        best_valid_loss = valid_loss
        torch.save(model.state_dict(), best_model_path)
        print("\tbest model saved")


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

    print(f"test loss: {total_loss}")

    rouge = load("rouge")
    results = rouge.compute(
        predictions=all_predictions,
        references=all_references,
    )
    if results is None:
        print("unable to calculate rouge score")
    else:
        print("rouge scores ->")
        for key, val in results.items():
            print(f"\t{key}: {val}")
