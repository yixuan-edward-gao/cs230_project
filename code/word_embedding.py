"""
Tests logistic regression, DNN, and 1D CNN on word embeddings.
Evaluates each model on a large dataset and a small dataset.

Usage:
python3 word_embedding.py

CS 230 Final Project
December 2, 2021
Edward Gao
"""
from utils import *
from torch_utils import *
import pandas as pd
import torch
import torch.nn as nn

class DNNModel(nn.Module):
    def __init__(self):
        super(DNNModel, self).__init__()
        self.layers = nn.Sequential(nn.Linear(300, 100),
                                    nn.BatchNorm1d(100),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(100, 50),
                                    nn.BatchNorm1d(50),
                                    nn.ReLU(),
                                    nn.Dropout(0.1),
                                    nn.Linear(50, 3))
    
    def forward(self, x):
        logits = self.layers(x)
        return logits

    
class WordEmbeddingDNNModel(MyModel):
    def __init__(self, data, partition):
        super(WordEmbeddingDNNModel, self).__init__(DNNModel(), data.iloc[:, :300], data['label'], 128, 
                                          partition, 0.0001)

        
class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.layers = nn.Sequential(nn.Conv1d(1, 10, 5),
                                    nn.ReLU(),
                                    nn.MaxPool1d(5),
                                    nn.Conv1d(10, 30, 5),
                                    nn.ReLU(),
                                    nn.MaxPool1d(5),
                                    nn.Flatten(),
                                    nn.Linear(330, 50),
                                    nn.ReLU(),
                                    nn.Linear(50, 20),
                                    nn.ReLU(),
                                    nn.Linear(20, 3))
    
    def forward(self, x):
        logits = self.layers(x)
        return logits

    
class WordEmbeddingConvModel(MyModel):
    def __init__(self, data, partition):
        super(WordEmbeddingConvModel, self).__init__(ConvModel(), data.iloc[:, :300], data['label'], 128, 
                                          partition, 0.00001, conv=True)

        
if __name__ == '__main__':
    large_data = pd.read_csv('../data/egfr_mol2vec.csv')
    small_data = pd.read_csv('../data/chymotrypsin_mol2vec.csv')
    
    print('Testing logistic regression on large dataset')
    test_model('logreg', None, 0.06, 1, 0, X=large_data.iloc[:, :300], y=large_data['label'])
    print('Testing logistic regression on small dataset')
    test_model('logreg', None, 0.2, 50, 0, X=small_data.iloc[:, :300], y=small_data['label'])
    
    print('Testing DNN model on large dataset')
    data = pd.read_csv('../data/egfr_filtered.csv')
    test_model(WordEmbeddingDNNModel, large_data, 0.06, 1, 500, print_loss=True, plot=False)
   
    print('Testing DNN model on small dataset')
    data = pd.read_csv('../data/chymotrypsin_filtered.csv')
    test_model(WordEmbeddingDNNModel, small_data, 0.2 , 50, 500, print_loss=False, plot=False)
    
    print('Testing 1D CNN model on large dataset')
    data = pd.read_csv('../data/egfr_filtered.csv')
    test_model(WordEmbeddingConvModel, large_data, 0.06, 1, 100, print_loss=True, plot=False)
   
    print('Testing 1D CNN model on small dataset')
    data = pd.read_csv('../data/chymotrypsin_filtered.csv')
    test_model(WordEmbeddingConvModel, small_data, 0.2 , 50, 100, print_loss=False, plot=False)