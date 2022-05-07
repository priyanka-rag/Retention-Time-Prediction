'''Author: Priyanka Raghavan
   Description: processing different types of splits'''

path = 'C:/Users/ChemeGrad2020/OneDrive/10.C51/Project'
from data_processing import init_data_processing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
import sys
import matplotlib.pyplot as plt

def get_scaffold_splits(): # organize the various clusters from each type of splits
    splits = pd.read_csv(path + '/data/scaffold_splits.csv')
    fgp_splits = splits['Kmeans from PCA']
    scaffold_splits = splits['Kmeans of Scaffold Split']

    # for the FGP Split, separate the split groupings into arrays
    df_list = [d for _, d in splits.groupby(['Kmeans from PCA'])]
    fgp_splits = [df.index.values for df in df_list]

    # for the Scaffold Split, separate the split groupings into arrays
    df_list = [d for _, d in splits.groupby(['Kmeans of Scaffold Split'])]
    scaffold_splits = [df.index.values for df in df_list]
    
    return fgp_splits,scaffold_splits

def get_model_input(split_type): # get input for train_model() method in models/vanillampnn.py from the splits
    inchi_list,pubchem_ids,smiles_list,fgp_list,retention_times = init_data_processing()
    fgp_splits,scaffold_splits = get_scaffold_splits()

    if split_type == 'fgp':
        train_idx, test_idx = fgp_splits[3].tolist(), fgp_splits[2].tolist() # group the splits
    if split_type == 'scaffold':
        train_idx, test_idx = scaffold_splits[0].tolist(), scaffold_splits[2].tolist() # group the splits
    
    train_smiles = [smiles_list[i] for i in range(len(smiles_list)) if i in train_idx]
    train_rt = [retention_times[i] for i in range(len(retention_times)) if i in train_idx]
    test_smiles = [smiles_list[i] for i in range(len(smiles_list)) if i in test_idx]
    test_rt = [retention_times[i] for i in range(len(retention_times)) if i in test_idx]
    
    return train_smiles,test_smiles,train_rt,test_rt

