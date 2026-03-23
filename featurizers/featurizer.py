
from typing import List

import numpy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pandas import Series
import torch

from utils import Indexer
import utils

class Featurizer:

    indexer: Indexer

    def __init__(self):
        self.indexer = Indexer()

    def to_feature(self, tokens: List[str]) -> torch.Tensor | str:
        utils.not_implemented()

    def build_vocab_repr(self, row: Series) -> Series:
        """
        Tokenize the words, split by punctuation, remove stop words, make all words lowercase
        """
        row["text"] = self.tokenize(row["text"])
        return row
    
    def tokenize(self, text: str) -> List[str]:
        words = word_tokenize(text)
        stop_words = set(stopwords.words('english'))
        stop_words.update(['[', ']'])
        final_words: List[str] = []
        for word in words:
            word = word.lower()
            if word not in stop_words:
                final_words.append(word)
        return final_words