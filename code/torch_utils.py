"""
Common functions and classes used for all pytorch models.

CS 230 Final Project
Edward Gao
December 2, 2021
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


class MolDataset(Dataset):
    """
    Dataset for molecular data
    """
    def __init__(self, X, y, X_func=lambda a: a.to_numpy(dtype=np.int32), X_get_func=lambda a: a):
        """
        Initiates a dataset.
        
        :param X: all training features
        :param y: all ground truth labels
        :param X_func: a function that converts X into the form it should be stored as
        :param X_get_func: a function that processes an element in X to the form that should be fed to a model as input
        """
        self.X = X_func(X)
        self.y = y.to_numpy(dtype=np.int32)
        self.f = X_get_func
    
    def __getitem__(self, index):
        return self.f(self.X[index]), self.y[index]
    
    def __len__(self):
        return self.y.shape[0]

    
def torch_accuracy(loader, model, device, conv=False):
    """
    Evaluates the accuracy of a pytorch model
    
    :param loader: a DataLoader instance
    :param model: a torch.nn.Module
    :param device: 'cuda' or 'cpu'
    :param conv: whether this model is a convolution model
    :return: the accuracy of the input model evaluated on the dataset in the dataloader
    """
    size = len(loader.dataset)
    correct = 0
    model.eval()
    with torch.no_grad():
        for X, y in loader:
            if conv:
                X = X.unsqueeze(1)
            X, y = X.to(device).float(), y.to(device).long()
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    return correct / size


def train(dataloader, model, loss_fn, optimizer, device, print_loss=True, conv=False):
    """
    Trains a pytorch model on a dataset.
    
    :param dataloader: a DataLoader instance with the data to be trained on
    :param model: a torch.nn.Module
    :param loss_fn: the loss function
    :param optimizer: the optimizer used for training
    :param device: 'cuda' or 'cpu'
    :param print_loss: whether to print the loss value during training
    :param conv: whether this model is a convolution model
    :return: a list of loss values calcualted during training
    """
    losses = []
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        if conv:
            X = X.unsqueeze(1)
        X, y = X.to(device).float(), y.to(device).long()
        pred = model(X)
        loss = loss_fn(pred, y)
        losses.append(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if print_loss and batch % 100 == 0:
            loss = loss.item()
            print(f'loss: {loss}')
    return losses


def init_weights(m):
    """
    He initialization for linear layers in a model
    
    :param m: a layer in a neural network
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        m.bias.data.fill_(0.01)

         
            
class MyModel:
    """
    Wrapper type for all neural network models used in this project.
    Specific models should subclass this class and specify parameters.
    """
    def __init__(self, model, X, y, batch_size, partition, learning_rate, reg_strength=0, seed=None, X_func=lambda a: a.to_numpy(dtype=np.int32), X_get_func=lambda a: a, conv=False):
        """
        Builds a neural network model.
        
        :param model: the model to be trained; an instance of a torch.nn.Module
        :param X: all input features (train and test sets)
        :param y: all ground truth labels (train and test sets)
        :param batch_size: mini batch size
        :param partition: ratio of the input dataset to use as the test set
        :param learning_rate: learning rate
        :param reg_strength: strength of L2 regularization
        :param seed: for controlling how the dataset is split into train and test sets
        :param X_func: a function that converts X into the form it should be stored as
        :param X_get_func: a function that processes an element in X to the form that should be fed to a model as input
        :param conv: whether this model is a convolution model
        """        
        
        # split into train and test sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=partition, random_state=seed)
        
        # initialize datasets and dataloaders
        self.train_dset = MolDataset(self.X_train, self.y_train, X_func, X_get_func)
        self.test_dset = MolDataset(self.X_test, self.y_test, X_func, X_get_func)
        self.train_loader = DataLoader(self.train_dset, batch_size=batch_size, shuffle=True)
        self.test_loader = DataLoader(self.test_dset, batch_size=batch_size, shuffle=True)
        
        # initialize model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.model.apply(init_weights)
        
        # loss function and optimizer
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate, weight_decay=reg_strength)
        
        self.conv = conv
        
    def train(self, epochs, print_loss=True):
        """
        Trains the model
        
        :param epochs: how many epochs to train for
        :param print_loss: whether to print the loss values during training
        :return: list of loss values, train accuracies, and test accuracies as training progressed
        """
        losses = []
        train_accuracies = []
        test_accuracies = []
        for t in range(epochs):
            if print_loss:
                print(f'Epoch {t + 1}\n--------------------')
            losses += train(self.train_loader, self.model, self.loss, self.optimizer, self.device, print_loss, self.conv)
            accuracies = self.evaluate()
            train_accuracies.append(accuracies[0])
            test_accuracies.append(accuracies[1])
        return losses, train_accuracies, test_accuracies
   
    def evaluate(self):
        """
        Evaluates the accuracy of the model
        
        :return: the current train and test accuracies
        """
        train_accuracy = torch_accuracy(self.train_loader, self.model, self.device, self.conv)
        test_accuracy = torch_accuracy(self.test_loader, self.model, self.device, self.conv)
        return train_accuracy, test_accuracy

    
def plot(data, xlabel, ylabel):
    """
    Plots a series of data
    
    :param data: a series of values to plot
    :param xlabel: label of x axis
    :param ylabel: label of y axis
    """
    plt.plot(list(range(len(data))), data)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()
    
    
def evaluate_model(model, epochs, print_loss=True, plot_metrics=False):
    """
    Trains a neural network and evaluates it accuracy
    
    :param model: the model to evaluate; an instance of MyModel
    :param epochs: how many epochs to train for
    :param print_loss: whether to print the loss values during training
    :param plot_metrics: whether to plot the loss and accuracy values
    :return: train and test accuracies of the model
    """
    losses, train_accuracies, test_accuracies = model.train(epochs, print_loss)
    train_accuracy, test_accuracy = model.evaluate()
    
    if plot_metrics:
        plot(losses, '# of iterations', 'loss')
        plot(train_accuracies, 'epoch', 'train accuracy')
        plot(test_accuracies, 'epoch', 'test accuracy')
        
    return train_accuracy, test_accuracy