"""
Tests logistic regression, DNN, and 1D CNN on molecular fingerprints.
Evaluates each model on a large dataset and a small dataset.

Usage:
python3 fingerprint.py

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
        self.layers = nn.Sequential(nn.Linear(2048, 500),
                                    nn.BatchNorm1d(500),
                                    nn.ReLU(),
                                    nn.Linear(500, 200),
                                    nn.BatchNorm1d(200),
                                    nn.ReLU(),
                                    nn.Linear(200, 3))
    
    def forward(self, x):
        logits = self.layers(x)
        return logits

    
class FingerprintDNNModel(MyModel):
    def __init__(self, data, partition):
        super(FingerprintDNNModel, self).__init__(DNNModel(), data['SMILES'], data['label'], 128, 
                                          partition, 0.0001, X_func=lambda a: a.tolist(), 
                                          X_get_func=lambda a: fingerprint(a))

        
class ConvModel(nn.Module):
    def __init__(self):
        super(ConvModel, self).__init__()
        self.layers = nn.Sequential(nn.Conv1d(1, 10, 7),
                                    nn.ReLU(),
                                    nn.MaxPool1d(10),
                                    nn.Conv1d(10, 30, 7),
                                    nn.ReLU(),
                                    nn.MaxPool1d(10),
                                    nn.Flatten(),
                                    nn.Linear(570, 50),
                                    nn.ReLU(),
                                    nn.Linear(50, 20),
                                    nn.ReLU(),
                                    nn.Linear(20, 3))
    
    def forward(self, x):
        logits = self.layers(x)
        return logits

    
class FingerprintConvModel(MyModel):
    def __init__(self, data, partition):
        super(FingerprintConvModel, self).__init__(ConvModel(), data['SMILES'], data['label'], 128, 
                                          partition, 0.001, X_func=lambda a: a.tolist(), 
                                          X_get_func=lambda a: fingerprint(a), conv=True)

        
if __name__ == '__main__':
    large_data = pd.read_csv('../data/egfr_filtered.csv')
    small_data = pd.read_csv('../data/chymotrypsin_filtered.csv')
    
    print('Testing logistic regression on large dataset')
    test_model('logreg', None, 0.06, 1, 0, X=np.array([fingerprint(i) for i in large_data['SMILES']]), y=large_data['label'])
    print('Testing logistic regression on small dataset')
    test_model('logreg', None, 0.2, 50, 0, X=np.array([fingerprint(i) for i in small_data['SMILES']]), y=small_data['label'])
    
    
    print('Testing DNN model on large dataset')
    data = pd.read_csv('../data/egfr_filtered.csv')
    test_model(FingerprintDNNModel, large_data, 0.06, 1, 10, print_loss=False, plot=False)
   
    print('Testing DNN model on small dataset')
    data = pd.read_csv('../data/chymotrypsin_filtered.csv')
    test_model(FingerprintDNNModel, small_data, 0.2 , 50, 10, print_loss=False, plot=False)
    
    print('Testing 1D CNN model on large dataset')
    data = pd.read_csv('../data/egfr_filtered.csv')
    test_model(FingerprintConvModel, large_data, 0.06, 1, 10, print_loss=False, plot=False)
   
    print('Testing 1D CNN model on small dataset')
    data = pd.read_csv('../data/chymotrypsin_filtered.csv')
    test_model(FingerprintConvModel, small_data, 0.2 , 50, 10, print_loss=False, plot=False)
