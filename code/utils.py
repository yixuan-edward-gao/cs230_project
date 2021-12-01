"""
Utility functions for data processing and model testing.

CS 230 Final Project
Edward Gao
December 2, 2021
"""

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import RDKFingerprint
from mol2vec.features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
from sklearn.linear_model import LogisticRegression
from torch_utils import *


def fingerprint(smiles):
    """
    Generates molecular fingerprint for a molecule.
    
    :param smiles: SMILES string of a molecule
    :return: the corresponding 2048-bit vector fingerprint, as a np array
    """
    mol = Chem.MolFromSmiles(smiles)
    return np.array(RDKFingerprint(mol))


def generate_word_embedding(data_file, out_file, model_file='../model_300dim.pkl'):
    """
    Uses a pre-trained model to generate word embeddings for a list of molecules.
    Prepends the embeddings as columns to the input data file 
    and writes an output csv file.
    
    :param data_file: path to a csv data file containing a 'SMILES' column of SMILES strings
    :param out_file: path to the output csv file
    :param model_file: path to a pre-trained model
    :return: data from the new csv file written to disk, as a pandas DataFrame
    """
    data = pd.read_csv(data_file)
    mol = [Chem.MolFromSmiles(i) for i in data['SMILES']]
    sentence = [MolSentence(mol2alt_sentence(i, radius=1)) for i in mol]
    w2v_model = word2vec.Word2Vec.load(model_file)
    embedding = [DfVec(x) for x in sentences2vec(sentence, w2v_model)]
    data_mol2vec = np.array([x.vec for x in embedding])
    data_mol2vec = pd.DataFrame(data_mol2vec)
    new_data = pd.concat([data_mol2vec, data], axis=1)
    new_data.to_csv(out_file, index=False)
    return new_data


# all characters used in SMILES strings
SMILES_CHARS = [' ',
                '#', '%', '(', ')', '+', '-', '.', '/',
                '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                '=', '@',
                'A', 'B', 'C', 'F', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P',
                'R', 'S', 'T', 'V', 'X', 'Z',
                '[', '\\', ']',
                'a', 'b', 'c', 'e', 'g', 'i', 'l', 'n', 'o', 'p', 'r', 's',
                't', 'u']
smi2index = dict( (c,i) for i,c in enumerate( SMILES_CHARS ) )
index2smi = dict( (i,c) for i,c in enumerate( SMILES_CHARS ) )


def smiles_encoder(smiles, maxlen=500):
    """
    Generates one-hot encodings of a SMILES string.
    Each column represents a character.
    Pads with columns of zeros to some maximum length.
    
    :param smiles: SMILES string of a molecule
    :param maxlen: maximum legnth allowed for SMILES strings; shorter strings are padded to this length
    :return: the one-hot encoding of the SMILES string as a np array
    """
    X = np.zeros((len(SMILES_CHARS), maxlen))
    for i, c in enumerate(smiles):
        X[smi2index[c], i] = 1
    return X


def smiles_decoder(X):
    """
    Converts a one-hot encoding of a SMILES string back to the original 1D string.
    
    :param X: one-hot encoding of a molecule
    :return: corresponding SMILES string
    """
    smi = ''
    X = X.argmax(axis=0)
    for i in X:
        smi += index2smi[i]
    return smi.strip()


def accuracy(truth, predicted):
    """
    Calculates the accuracy of the predictions of a classifier
    
    :param truth: list of ground truth labels
    :param predicted: list of predicted labels
    :returns: accuracy of the prediction
    """
    return np.sum(predicted == truth) / len(predicted)


def test_log_reg(X, y, partition):
    """
    Runs logistic regression on a dataset and evaluates its accuracy.
    
    :param X: all input features (train and test sets)
    :param y: all ground truth labels (train and test sets)
    :param partition: percentage of the input dataset to be used as the test set
    :return: train and test accuracies of logsitic regression
    """
    clf = LogisticRegression(max_iter=1000000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=partition)
    clf.fit(X_train, y_train)
    train_preds = clf.predict(X_train)
    test_preds= clf.predict(X_test)
    return accuracy(y_train, train_preds), accuracy(y_test, test_preds)


def test_model(model, data, partition, runs, epochs, print_loss=True, plot=False, X=None, y=None):
    """
    Tests a classifier model on a dataset and evaluates its accuracy.
    Prints out the train and test accuracies.
    
    :param model: a subclass of torch_utils.MyModel; this object will be called to construct a DL model instance; or 'logreg' for logistic regression
    :param data: the entire dataset (train and test); not used if model is 'logreg'
    :param partition: percentage of the input dataset to be used as the test set
    :param runs: how many times the model is evaluated; the reported accuracies are averaged across all runs
    :param epochs: number of training epochs; not used if model is 'logreg'
    :param print_loss: whether to print loss values during training; not used if model is 'logreg'
    :param plot: whether to plot loss and accuracy values after each run; not used if model is 'logreg'
    :param X: all input features (train and test sets); only used if model is 'logreg'
    :param y: all ground truth labels (train and test sets); only used if model if 'logreg'
    :return: train and test accuracies of the model averaged across all runs
    """
    train_accuracies, test_accuracies = [], []
    
    for _ in range(runs):
        if model == 'logreg':
            train_accuracy, test_accuracy = test_log_reg(X, y, partition)
        else:
            m = model(data, partition)
            train_accuracy, test_accuracy = evaluate_model(m, epochs, print_loss, plot_metrics=plot)
        
        train_accuracies.append(train_accuracy)
        test_accuracies.append(test_accuracy)
        
    train_mean = np.mean(train_accuracies)
    test_mean = np.mean(test_accuracies)
    print(f'Train accuracy mean: {train_mean}')
    print(f'Test accuracy mean: {test_mean}')
        
    return train_mean, test_mean