"""
Preprocess data downloaded from ChEMBL, perform data augmentation, and generate word embeddings
Usage: python3 preprocess.py

CS 230 Final Project
Edward Gao
December 2, 2021
"""
import pandas as pd
import os
from sklearn.utils import shuffle
from utils import generate_word_embedding

def load_molecules():
    """
    Reads in a large list of random molecules from a dataset downloaded from ChEMBL.
    The file path is hard coded.
    Saves the data file if not already present.
    
    :return: a pandas DataFrame of SMILES strings of a large number of random molecules
    """
    if os.path.exists('../data/molecules_filtered.csv'):
        return pd.read_csv('../data/molecules_filtered.csv')
    else:
        molecules = pd.read_csv('../data/molecules.tsv', sep='\t')
        molecules = molecules['Smiles'].astype({'Smiles': 'string'}).rename('SMILES').to_frame()
        molecules = molecules[molecules['SMILES'].notna()]
        molecules.to_csv('../data/molecules_filtered.csv', index=False)
        return molecules

    
def read_file(path, activity_type, activity_unit):
    """
    Reads in some data file
    
    :param path: path to data file (tsv downloaded from ChEMBL)
    :param activity_type: name of biological activity to process
    :param activity_unit: unit of bioactivity measurement
    :return: pandas DataFrame consisting of SMILES and bioactivity values
    """
    data = pd.read_csv(path, sep='\t')
    data = data[data['Standard Units'] == activity_unit]
    assert((data['Standard Type'] == activity_type).all())
    
    data = data[['Smiles', 'Standard Value']].astype({'Smiles': 'string', 'Standard Value': 'float64'})
    data = data.rename(columns={'Smiles': 'SMILES', 'Standard Value': activity_type})
    
    # remove entries with missing values
    data = data[data[activity_type].notna()]
    data = data[data['SMILES'].notna()]
    
    print(f'Read in data file {path}. There are {data.shape[0]} valid molecules.')
    print('Here are the first few lines:')
    print(data.head())
    
    return data


def count(data, activity_type, thresholds):
    """
    Counts how many of the input molecules belongs to each specified category.
    For example, if thresholds is [a, b, c, d], counts the number of molecules
    whose activity value falls in:
    - (-inf, a)
    - [a, b)
    - [b, c)
    - [c, d)
    - [d, +inf)
    
    :param data: pandas DataFrame containing molecules and their bioactivity data
    :param activity_type: name of bioactivity
    :param thresholds: threshold values of bioactivity used for classification purposes; must be in ascending order
    :return: list of numbers of molecules that belong in each bin, as specified in thresholds
    """
    counts = [0 for _ in range(len(thresholds) + 1)]
    counts[0] = (data[activity_type] < thresholds[0]).sum()
    print('Number of molecules with:')
    print(f'{activity_type} < {thresholds[0]}:\t\t{counts[0]}')      
    
    for i in range(1, len(thresholds)):
        counts[i] = ((data[activity_type] >= thresholds[i - 1]) & (data[activity_type] < thresholds[i])).sum()
        print(f'{thresholds[i - 1]} <= {activity_type} < {thresholds[i]}:\t{counts[i]}')
    
    counts[-1] = (data[activity_type] >= thresholds[-1]).sum()
    print(f'{activity_type} >= {thresholds[-1]}:\t\t{counts[-1]}')
    
    return counts


def augment_data(num, val, activity_type):
    """
    Performs data augmentation by picking some random molecules.
    The assumption is that random molecules are very unlikely to have significant
    bioactivity toward a given protein, so all molecules are assigned the same
    bioactivity value.
    
    :param num: number of random molecules to fetch
    :param val: bioactivity value assigned to the selected random molecules
    :param activity_type: name of bioactivity
    :return: a pandas DataFrame containing random molecules
    """
    molecules = load_molecules()
    print(f'Number of random molecules in dataset: {len(molecules)}')
    supp = molecules.sample(num)
    print(f'Selected {len(supp)} random molecules')
    supp[activity_type] = val
    
    return supp


def assign_label(val, thresholds):
    """
    Assigns a label to a molecule based on its bioactivity.
    
    :param val: bioactivity value of a molecule
    :param thresholds: bioactivity thresholds used to characterize a molecule
    :return: the assigned label (as a non-negative integer)
    """
    for i, n in enumerate(thresholds):
        if val < n:
            return i
    return len(thresholds)
    
    
def preprocess(path, out, activity_type='IC50', activity_unit='nM', thresholds=[100, 10000], augment='last'):
    """
    Preprocesses data and optionally perform data augmentation.
    Saves the processed dataset.
    
    :param path: path to data file; must be tsv downloaded from ChEMBL
    :param out: path of output data file
    :param activity_type: name of bioactivity; must be one of the names used in ChEMBL; default IC50
    :param activity_unit: unit of bioactivity; must be one of the options used in ChEMBL; default nM
    :param thresholds: bioactivity thresholds used to characterize a molecule; default [100, 10000]
    :param augment: whether to perform data augmentation; must be None, 'first', or 'last';
                    if None, no augmentation is performed;
                    if 'first', augments the first category (those with activities < thresholds[0];
                    if 'last', augments the last category (those with activities >= thresholds[-1]
    :return: pandas DataFrame containing molecules with their SMILES strings and assigned labels
    """
    data = read_file(path, activity_type, activity_unit) 
    counts = count(data, activity_type, thresholds)
        
    if augment is not None:
        num, val = 0, 0
        # perform data augmentation so that the number of examples in the augmented category
        # is equal to the average number of examples in all other categories
        if augment == 'first':
            val = thresholds[0] - 1
            num = int(sum(counts[1:]) / (len(counts) - 1) - counts[0])
        elif augment == 'last':
            val = thresholds[-1] + 1
            num = int(sum(counts[:-1]) / (len(counts) - 1) - counts[-1])
            
        supp = augment_data(num, val, activity_type)
        data = data.append(supp)
            
    data['label'] = list(map(lambda x: assign_label(x, thresholds), data[activity_type]))
        
    data = shuffle(data)
    data.reset_index(inplace=True, drop=True)
        
    print('Successfully processed data file. Here are the first few lines:')
    print(data.head())
        
    data.to_csv(out, index=False)
    print(f'Saved data file to {out}')
        
    return data
            

if __name__ == '__main__':
    # filter data files downloaded from ChEMBL, perform augmentation, and generate word embeddings
    preprocess('../data/egfr.tsv', '../data/egfr_filtered.csv')
    preprocess('../data/chymotrypsin.tsv', '../data/chymotrypsin_filtered.csv', thresholds=[500, 10000], augment=None)
    generate_word_embedding('../data/egfr_filtered.csv', '../data/egfr_mol2vec.csv')
    generate_word_embedding('../data/chymotrypsin_filtered.csv', '../data/chymotrypsin_mol2vec.csv')