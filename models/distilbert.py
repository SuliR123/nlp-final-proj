from typing import List

import pandas as pd
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from models.model import Model


class DistilBert(Model):

    def __init__(
        self,
        labels: List[str],
        lr: float = 5e-5,
        epochs: int = 3,
        batch_size: int = 16,
        max_length: int = 128,
        pretrained: str = "distilbert-base-uncased",
    ):
        super().__init__()
        self.labels = sorted(labels)
        self.label_to_idx = {l: i for i, l in enumerate(self.labels)}
        self.idx_to_label = {i: l for i, l in enumerate(self.labels)}
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.max_length = max_length

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.tokenizer = DistilBertTokenizerFast.from_pretrained(pretrained)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            pretrained, num_labels=len(self.labels)
        ).to(self.device)

    def fit(self, train_df: pd.DataFrame):
        texts = train_df["text"].tolist()
        label_ids = [self.label_to_idx[l] for l in train_df["label"].tolist()]
        print(f"Tokenizing {len(texts)} examples...")
        dataset = _TextDataset(texts, label_ids, self.tokenizer, self.max_length)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        self.model.train()
        for epoch in range(self.epochs):
            total_loss = 0.0
            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for batch in pbar:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                optimizer.zero_grad()
                outputs = self.model(**batch)
                outputs.loss.backward()
                optimizer.step()
                total_loss += outputs.loss.item()
                pbar.set_postfix(loss=f"{outputs.loss.item():.4f}")
            print(f"Epoch {epoch + 1}/{self.epochs} avg loss: {total_loss / len(loader):.4f}")

    def predict(self, text: str) -> str:
        self.model.eval()
        enc = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**enc).logits
        return self.idx_to_label[logits.argmax(dim=-1).item()]


class _TextDataset(Dataset):

    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
