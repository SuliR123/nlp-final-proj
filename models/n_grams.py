
from typing import Dict, List

import numpy as np
from pandas import Series

from featurizers.featurizer import Featurizer
from models.model import Model


class NGrams(Featurizer, Model):

    # for each class, count the appearences of bigrams in it
    class_corpuses : Dict[str, Dict[str, int]]
    n: int # number of words to look back in context
    interpolation: List[float]
    unigram_count: int
    k: float

    def __init__(self, n: int, k: float):
        super().__init__()
        self.class_corpuses = {}
        self.n = n
        self.interpolation = [.1, .3, .6] # TODO, automate creating the weight for any n
        self.unigram_count = 0
        self.k = k
        self.global_vocab = set() 
        self.class_total_words = {}

    def predict(self, text: str):
        # add padding to the sentence
        tokens = self.tokenize(text)
        tokens = ["[start]"] * (self.n - 1) + tokens + ["[end]"]
        pred = None
        best_score = None
        for label in self.class_corpuses:
            prob = self.calculate_prob(tokens, label)
            if best_score is None or prob > best_score:
                best_score = prob
                pred = label
        return pred
    
    def calculate_prob(self, tokens: List[str], label: str):
        count_dict = self.class_corpuses[label]
        vocab_size = len(self.global_vocab) # Uniform across all classes
        final_prob = 0
        
        for i, token in enumerate(tokens):
            if token == "[start]": continue
            
            total_prob_for_n_gram = 0
            for curr_n in range(1, self.n + 1):
                n_gram_str = self.to_n_gram_str(curr_n, tokens, i)
                count = self.get_count(count_dict, n_gram_str)
                
                if curr_n == 1:
                    # Use class-specific total words, not global
                    prob = self.calc_prob_laplace(count, self.class_total_words[label], vocab_size)
                else:
                    prev_n_gram_str = self.to_n_gram_str(curr_n - 1, tokens, i - 1)
                    prev_count = self.get_count(count_dict, prev_n_gram_str)
                    prob = self.calc_prob_laplace(count, prev_count, vocab_size)
                
                total_prob_for_n_gram += self.interpolation[curr_n - 1] * prob
            
            final_prob += np.log(total_prob_for_n_gram)
            
        # Optional: Add Log Prior P(Class) to handle class imbalance
        # final_prob += np.log(self.class_total_words[label] / sum(self.class_total_words.values()))
        
        return final_prob

    def calc_prob_laplace(self, count: int, total: int, vocab_size: int):
        numerator = count + self.k
        denom = total + self.k * vocab_size
        return numerator / denom

    def get_count(self, count_dict: Dict[str, int], s: str):
        if s in count_dict:
            return count_dict[s]
        else:
            return 0
    
    def build_vocab_repr(self, row: Series) -> Series:        
        label = row["label"]
        if label not in self.class_corpuses:
            self.class_corpuses[label] = {}
            self.class_total_words[label] = 0
        
        count_dict = self.class_corpuses[label]
        row = super().build_vocab_repr(row) 
        tokens = ["[start]"] * (self.n - 1) + row["text"] + ["[end]"]
        
        for i, token in enumerate(tokens):
            if token == '[start]': continue
            
            # Track class total and global vocabulary for unigrams
            self.class_total_words[label] += 1
            self.global_vocab.add(token) 
            
            for current_n in range(1, self.n + 1):
                n_gram = self.to_n_gram_str(current_n, tokens, i)
                count_dict[n_gram] = count_dict.get(n_gram, 0) + 1
        return row
    
    def to_n_gram_str(self, current_n: int, tokens: List[str], i: int):
        return f"{current_n}gram:{" ".join(tokens[i - (current_n - 1): i + 1])}"