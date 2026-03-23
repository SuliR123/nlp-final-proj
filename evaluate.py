
import argparse
import json
from pathlib import Path
import kagglehub
import nltk
import pandas as pd

from featurizers.featurizer import Featurizer
from models.model import Model
from models.n_grams import NGrams
from sklearn.model_selection import train_test_split

def _parse_args() -> argparse.Namespace:
    """
    Command-line arguments to the system.
    """
    parser = argparse.ArgumentParser(description='Argument parsing for evaluating models')
    parser.add_argument('--model', type=str, default='n-grams', choices=['n-grams', 'fixed-window', 'bert'], help='Possible models to train')
    parser.add_argument('--training_mode', type=str, default='development', choices=['development', 'prod'], help='Training on portion of dataset or entire dataset')
    parser.add_argument('--download_nltk', action='store_true', default=False)
    args = parser.parse_args()
    return args

class EvalModel():

    training_data: pd.DataFrame

    def __init__(self):
        pass

    def load_data(self, featurizer: Featurizer, model: str, batch_size: int = 16, training: bool = False) -> Model:
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
        
        df = df.loc[df['example_very_unclear'] == False, filter_cols]
        df["label"] = df[filter_cols[1:]].idxmax(axis=1)

        
        train, test = train_test_split(df, test_size=.2)
        train.apply(featurizer.build_vocab_repr, axis = 1)

        print("=====ALL COLUMNS======")
        print(train.columns)
        print("=====EXAMPLE ROWS=====")
        print(train.head())
        print("=====TRAIN/VALIDATION SET METRICS=====")
        print(f"LENGTH OF TRAINING {len(train)}")
        print(f"LENGTH OF VALDATION {len(test)}")

        return self.create_model(model, featurizer), train, test 

    def create_model(self, model: str, featurizer: Featurizer):
        if model == "n-grams":
            return featurizer
        else:
            raise Exception("Unknown model")
    
    def run_model_on_set(self, model: Model, validation_set: pd.DataFrame):
        validation_set["model_pred"] = validation_set["text"].map(model.predict)
        print(validation_set.head(n=10))
        accuracy = (validation_set["label"] == validation_set["model_pred"]).sum() / len(validation_set)
        return accuracy

if __name__ == "__main__":
    args = _parse_args()

    if args.download_nltk:
        if not nltk.download('stopwords'):
            raise Exception("Unable to download stopwords dataset")
        if not nltk.download('punkt_tab'):
            raise Exception("Unable to get tokenizer from model")

    eval_model = EvalModel()

    model, train, val_set = eval_model.load_data(NGrams(3, 1), args.model)
    print(f"ACCURACY ON TRAINING SET: {eval_model.run_model_on_set(model, train)}")
    print(f"ACCURACY ON DEV SET: {eval_model.run_model_on_set(model, val_set)}")

    

    
    