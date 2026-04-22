# NLP Final Project

# Steps To Run

## Install Required Dependencies
``` bash
pip3 install -r requirements.txt
```

Link to the dataset https://www.kaggle.com/datasets/debarshichanda/goemotions

## Run Project
For this project we have 3 models to run with some tweakable hyperparameters.

When running the project for the first time please include the --download_nltk flag to install the necessary data to run the project.

``` bash
python3 evaluate.py --download_nltk
```

The model flag lets you choose which model to run the program on. The choices are n-grams, fixed-window, and bert. By default the evaluate.py file will run the n-grams model.

### N-Grams
To run n-grams (only n-grams configuration is n=3 k=1) please use the following command

```bash
python3 evaluate.py 
```

This will run n-grams and evaluate the model on our kaggle training and validation datasets. The results of the model will be printed to the terminal.

### Fixed-Window
To run the fixed-window model please use the following command

``` bash
python3 evaluate.py --model fixed-window
```

Optionally you can also change the window size and word embedding dimension for the model with the flags:
--window_size (int)
--embedding_dim (int)

This command will run the fixed_window model for 10 epochs, each epoch will print the training loss, development loss, and perplexity. It will also create to plots that demonstrate these metrics over the epochs in the base directory of the project.

Our best model was run with the following command
``` bash
python3 evaluate.py --model fixed-window --window 5
```

### Bert
To run the bert model please use the following command

``` bash
python3 evaluate.py --model bert
```

Optionally you can specify the number of epochs and learning rate with the following flags
--epochs (int)
--lr (float)

When running bert, the model will print the epoch | training loss | and validation accuracy as the epochs go on

Our best model was ran with the following commmands 

``` bash
python3 evaluate.py --model bert --epochs 3 --lr 5e-6
```

Running this will result in a plot demonstrating the loss, validation accuracy, and perplexity over epochs.