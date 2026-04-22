

from typing import List

from pandas import Series

from featurizers.featurizer import Featurizer

import torch
import torch.nn as nn


class WordEmbeddings(Featurizer):

    embedding_dim: int
    embeddings: nn.Embedding | None
    window_size: int
    MAX_LENGTH = 30
    
    def __init__(self, embedding_dim: int, window_size: int):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.indexer.add_and_get_index("<PAD>")
        self.indexer.add_and_get_index("<UNK>")
        self.embeddings = None

    def to_feature(self, text: str):
        if len(self.indexer) == 0:
            raise Exception("No vocabulary initialized")

        if self.embeddings == None: # initialize embeddings dimension if it doesn't exist already
            self.embeddings = nn.Embedding(
                num_embeddings=len(self.indexer),
                embedding_dim=self.embedding_dim,
                padding_idx=0,
            )

        # create window
        indices = self.sentence_to_tensor(text)
        windows = self.create_windows(indices, self.window_size)
        
        return windows
    
    def build_vocab_repr(self, row) -> Series:
        words = self.tokenize(row["text"])
        for word in words:
            self.indexer.add_and_get_index(word)
        row["text"] = words
        return row
    
    def sentence_to_tensor(self, text: str) -> torch.Tensor:
        tokens = self.tokenize(text)
        if len(tokens) < self.MAX_LENGTH:
            tokens = tokens + ["<PAD>"] * (self.MAX_LENGTH - len(tokens))
        elif len(tokens) > self.MAX_LENGTH:
            tokens = tokens[:self.MAX_LENGTH]
        # convert words to indices, use <UNK> index for unknown words
        indices = [self.indexer.index_of(word) if self.indexer.contains(word) else 1 for word in tokens]
        return torch.tensor(indices)
    
    def create_windows(self, indices: torch.Tensor, window_size: int, pad_idx=0):
        # pad on both sides so every word gets a centered window
        padding = [pad_idx] * (window_size // 2)
        padded = padding + indices.tolist() + padding
        windows = []
        for i in range(len(indices)):
            window = padded[i:i + window_size]
            windows.append(window)
        return torch.tensor(windows) 