

from typing import List

from torch import nn
import torch

from featurizers.word_embeddings import WordEmbeddings
from models.model import Model
from utils import Indexer


class FixedWindow(Model):

    def __init__(self, fixed_window, featurizer: WordEmbeddings, label_indexer: Indexer):
        super().__init__()
        self.fixed_window = fixed_window
        self.featurizer = featurizer
        self.label_indexer = label_indexer

    def predict(self, text: str) -> str:
        self.fixed_window.eval()
        with torch.no_grad():
            logits = self.fixed_window.forward(self.featurizer.to_feature(text).unsqueeze(0))
            predicted_index = logits.argmax(dim=-1).item()
            return self.label_indexer.get_object(predicted_index)
        

    def predict_batch(self):
        return super().predict_batch()
    
    def get_metrics(self):
        return super().get_metrics()
    
    def plot(self):
        return super().plot()
    
class FixedWindowModel(nn.Module):

    window_size: int 
    embedding_dim: int
    num_classes: int
    hidden_dims: List[int]
    embedding: nn.Embedding
    linears: nn.ModuleList

    def __init__(self, window_size: int, embedding_dim: int, num_classes: int, hidden_dims: List[int], embedding: nn.Embedding):
        super().__init__()
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.hidden_dims = hidden_dims
        self.embedding = embedding
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(p=0.4)

        # initialize hidden layers
        hidden_dims.insert(0, window_size * embedding_dim)
        hidden_dims.append(num_classes)
        self.linears = nn.ModuleList([nn.Linear(hidden_dims[i - 1], hidden_dims[i]) for i in range(1, len(hidden_dims))])

    def forward(self, x):
        # x shape: (batch_size, num_windows, window_size)
        batch_size, num_windows, window_size = x.shape
        
        # embed all windows at once
        embedded = self.embedding(x)
        # (batch_size, num_windows, window_size, embedding_dim)
        
        # flatten each window into one vector
        embedded = embedded.view(batch_size, num_windows, -1)
        # (batch_size, num_windows, window_size * embedding_dim)
        
        # pass every window through the network simultaneously
        logits = embedded
        for i, l in enumerate(self.linears):
            logits = l(logits)
            if i != len(self.linears) - 1:
                logits = self.dropout(self.activation(logits))
        # (batch_size, num_windows, num_classes)
        
        # max pool across all windows — take the strongest signal per feature
        pooled, _ = logits.max(dim=1)
        # (batch_size, num_classes)
        
        return pooled
        