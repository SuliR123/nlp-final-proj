
import argparse
from pathlib import Path
from typing import Tuple
import kagglehub
from matplotlib import pyplot as plt
import nltk
import pandas as pd
from torch import nn, optim
import torch

from featurizers.featurizer import Featurizer
from featurizers.word_embeddings import WordEmbeddings
from models.fixed_window import FixedWindow, FixedWindowModel
from models.distilbert import DistilBert
from models.model import Model
from models.n_grams import NGrams
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.manifold import TSNE

from utils import Indexer

def _parse_args() -> argparse.Namespace:
    """
    Command-line arguments to the system.
    """
    parser = argparse.ArgumentParser(description='Argument parsing for evaluating models')
    parser.add_argument('--model', type=str, default='n-grams', choices=['n-grams', 'fixed-window', 'bert'], help='Possible models to train')
    parser.add_argument('--training_mode', type=str, default='development', choices=['development', 'prod'], help='Training on portion of dataset or entire dataset')
    parser.add_argument('--download_nltk', action='store_true', default=False)
    parser.add_argument('--window_size', type=int, default=3, help='Size of the window for the fixed winow ffnn')
    parser.add_argument('--embedding_dim', type=int, default=50, help="Size of the word embeddings")
    parser.add_argument('--epochs', type=int, default=3, help="Number of epochs to run bert on")
    parser.add_argument('--lr', type=float, default=.01, help="Learning rate for bert")
    args = parser.parse_args()
    return args

class EvalModel():

    training_data: pd.DataFrame
    label_to_index: Indexer

    def __init__(self):
        self.label_to_index = Indexer()

    def load_data(self, featurizer: Featurizer, model: str, batch_size: int = 16, training: bool = False) -> Tuple[Model, pd.DataFrame, pd.DataFrame]:
        """
        Load the kaggle dataset and initialize the featurizer to be used for given model
        Return untrained model
        """
        path: Path = Path(kagglehub.dataset_download("debarshichanda/goemotions")) / "data" / "full_dataset" / "goemotions_1.csv"

        print("Path to dataset files:", path)

        df = pd.read_csv(path)

        filter_cols = ['text', 'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion',
                    'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust',
                    'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy',
                    'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief',
                    'remorse', 'sadness', 'surprise', 'neutral']
        
        for label in filter_cols[1:]:
            self.label_to_index.add_and_get_index(label)
        
        df = df.loc[df['example_very_unclear'] == False, filter_cols]
        df["label"] = df[filter_cols[1:]].idxmax(axis=1)

        
        train, test = train_test_split(df, test_size=.2) 
        print(train.apply(featurizer.build_vocab_repr, axis = 1))

        print("=====ALL COLUMNS======")
        print(train.columns)
        print("=====EXAMPLE ROWS=====")
        print(train.head())
        print("=====TRAIN/VALIDATION SET METRICS=====")
        print(f"LENGTH OF TRAINING {len(train)}")
        print(f"LENGTH OF VALDATION {len(test)}")

        return train, test 

    def create_model(self, model: str, featurizer: Featurizer, df: pd.DataFrame):
        if model == "n-grams":
            return featurizer
        elif model == "fixed-window":
            return FixedWindow()
        elif model == "bert":
            return DistilBert(labels=df["label"].unique().tolist())
        else:
            raise Exception("Unknown model")
    
    def run_model_on_set(self, model: Model, validation_set: pd.DataFrame):
        validation_set["model_pred"] = validation_set["text"].map(model.predict)
        print(validation_set.head(n=10))
        
        accuracy = (validation_set["label"] == validation_set["model_pred"]).sum() / len(validation_set)
        
        labels = validation_set["label"]
        preds = validation_set["model_pred"]
        classes = labels.unique()
        
        f1_scores = []
        for c in classes:
            tp = ((labels == c) & (preds == c)).sum()
            fp = ((labels != c) & (preds == c)).sum()
            fn = ((labels == c) & (preds != c)).sum()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1)
        
        macro_f1 = sum(f1_scores) / len(f1_scores)
        
        return accuracy, macro_f1
    
    def train_fixed_window(self, train_data: pd.DataFrame, validation_data: pd.DataFrame, embedding_dim: int, featurizer: WordEmbeddings, window_size: int, development: bool = False) -> FixedWindow:
        x_tensor = torch.stack([featurizer.to_feature(text) for text in train_data["text"]])
        y_tensor = torch.tensor([self.label_to_index.index_of(label) for label in train_data["label"]])

        dataset = TensorDataset(x_tensor, y_tensor)
        loader = DataLoader(dataset, batch_size=32, shuffle=True)

        val_x_tensor = torch.stack([featurizer.to_feature(text) for text in validation_data["text"]])
        val_y_tensor = torch.tensor([self.label_to_index.index_of(label) for label in validation_data["label"]])

        val_dataset = TensorDataset(val_x_tensor, val_y_tensor)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

        loss = nn.CrossEntropyLoss()

        fixed_window_network = FixedWindowModel(window_size, embedding_dim, len(self.label_to_index), [256, 128], featurizer.embeddings)
        optimizer = optim.Adam(fixed_window_network.parameters(), lr=1e-3)

        loss_vs_epochs = []
        perplexity_vs_epochs = []
        validation_loss_vs_epochs = []

        for epoch in range(10):
            fixed_window_network.train()
            total_loss = 0
            
            for x_batch, y_batch in loader:
                # Zero out gradients
                optimizer.zero_grad()
                
                # Forward pass
                logits = fixed_window_network.forward(x_batch)
                
                # Compute loss
                batch_loss = loss(logits, y_batch)
                
                # Backward pass
                batch_loss.backward()
                
                # Update weights
                optimizer.step()
                
                total_loss += batch_loss.item()

            # get validation metrics for the current epoch
            fixed_window_network.eval()
            val_loss = 0
            with torch.no_grad():
                for x_batch, y_batch in val_loader:
                    logits = fixed_window_network.forward(x_batch)
                    batch_loss = loss(logits, y_batch)
                    val_loss += batch_loss.item()

            avg_loss = total_loss / len(loader)
            perplexity = torch.exp(torch.tensor(avg_loss))
            val_loss = val_loss / len(val_loader)
            val_eval_formatted = f"| Validation Loss: {val_loss:.4f}" if development else ""
            print(f"Epoch {epoch + 1} | Loss: {avg_loss:.4f} | Perplexity: {perplexity:.4f} {val_eval_formatted}")

            # store info to create plots
            loss_vs_epochs.append(avg_loss)
            perplexity_vs_epochs.append(perplexity)
            validation_loss_vs_epochs.append(val_loss)

        # plot the evaluation metrics vs epochs
        epochs = range(1, len(loss_vs_epochs) + 1)

        # plot losses 
        plt.plot(epochs, loss_vs_epochs, label="Train Loss")
        plt.plot(epochs, validation_loss_vs_epochs, label="Validation Loss")
        plt.title("Loss over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig("fixed_window_loss.png")

        plt.clf()

        # plot perplexity
        plt.plot(epochs, perplexity_vs_epochs, label="Perplexity")
        plt.title("Perplexity over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Perplexity")
        plt.legend()
        plt.savefig("fixed_window_perplexity.png")

        self.plot_embeddings(featurizer)

        return FixedWindow(fixed_window_network, featurizer, self.label_to_index)

    def plot_embeddings(self, featurizer, num_words=200):
        # get the embedding weights
        weights = featurizer.embeddings.weight.detach().numpy()
        
        # only plot the most common words to keep it readable
        weights = weights[:num_words]
        words = [featurizer.indexer.get_object(i) for i in range(num_words)]
        
        # reduce to 2D using t-SNE
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        reduced = tsne.fit_transform(weights)
        
        # plot
        plt.figure(figsize=(16, 12))
        plt.scatter(reduced[:, 0], reduced[:, 1], alpha=0.5)
        
        # label each point with the word
        for i, word in enumerate(words):
            plt.annotate(word, (reduced[i, 0], reduced[i, 1]), fontsize=8)
        
        plt.title("Word Embeddings Visualized with t-SNE")
        plt.savefig("embeddings_plot.png")
        plt.clf()

if __name__ == "__main__":
    args = _parse_args()

    if args.download_nltk:
        if not nltk.download('stopwords'):
            raise Exception("Unable to download stopwords dataset")
        if not nltk.download('punkt_tab'):
            raise Exception("Unable to get tokenizer from model")  

    eval_model = EvalModel()

    model = None
    if args.model == "n-grams":
        model = NGrams(3, 1)
        train, val_set = eval_model.load_data(model, args.model)
    elif args.model == "fixed-window":
        print(f"Fixed Window Model | Window Size {args.window_size} | Training Mode: {args.training_mode} | Embedding Size: {args.embedding_dim}")
        featurizer = WordEmbeddings(args.embedding_dim, args.window_size)
        train, val_set = eval_model.load_data(featurizer, args.model)
        model = eval_model.train_fixed_window(train, val_set, args.embedding_dim, featurizer, args.window_size, args.training_mode == 'development')
    elif args.model == "bert":
        print(f"Bert | Num epochs: {args.epochs} | LR: {args.lr}")
        train, val_set = eval_model.load_data(Featurizer(), args.model)
        model = DistilBert(labels=train["label"].unique().tolist(), epochs=args.epochs, lr=args.lr)
        model.fit(train, val_set)
    else:
        raise Exception(f"Must pass in either n-grams, fixed-window, or bert. Got {args.model}")
    
    acc, f1 = eval_model.run_model_on_set(model, train)
    print(f"ACCURACY ON TRAINING SET: {acc} | MACRO F1: {f1}")
    acc, f1 = eval_model.run_model_on_set(model, val_set)
    print(f"ACCURACY ON VALIDATION SET: {acc} | MACRO F1: {f1}")

    

    
    