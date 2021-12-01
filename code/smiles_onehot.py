"""
Tests a 2D CNN model on one-hot encodings of SMILES strings.
Evaluates model on a large dataset and a small dataset.

Usage:
python3 smiles_onehot.py

CS 230 Final Project
December 2, 2021
Edward Gao
"""
from utils import *
from torch_utils import *
import pandas as pd
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(1, 10, 3),
                                    nn.ReLU(),
                                    nn.MaxPool2d(5),
                                    nn.Conv2d(10, 30, 3),
                                    nn.ReLU(),
                                    nn.MaxPool2d(5),
                                    nn.Flatten(),
                                    nn.Linear(570, 50),
                                    nn.ReLU(),
                                    nn.Linear(50, 20),
                                    nn.ReLU(),
                                    nn.Linear(20, 3))
    
    def forward(self, x):
        logits = self.layers(x)
        return logits

class OneHotModel(MyModel):
    def __init__(self, data, partition):
        super(OneHotModel, self).__init__(Model(), data['SMILES'], data['label'], 128, 
                                          partition, 0.001, X_func=lambda a: a.tolist(), 
                                          X_get_func=lambda a: smiles_encoder(a), conv=True)
    

if __name__ == '__main__':
    large_data = pd.read_csv('../data/egfr_filtered.csv')
    small_data = pd.read_csv('../data/chymotrypsin_filtered.csv')
    print('Testing model on large dataset')
    test_model(OneHotModel, large_data, 0.06, 1, 10, True)
    
    print('Testing model on small dataset')
    test_model(OneHotModel, small_data, 0.2 , 50, 1, True)
    