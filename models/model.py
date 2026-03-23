

from typing import Tuple

import utils


class Model():
    
    def __init__(self):
        pass

    def predict(self, text: str) -> str:
        """
        Given input text, return the predicted label for the sample as a string
        """
        utils.not_implemented()

    def predict_batch(self):
        utils.not_implemented()

    def get_metrics(self) -> Tuple[float, float, float, float]:
        utils.not_implemented()

    def plot(self):
        utils.not_implemented()