from utils import read_data, get_fingerprints, scale_split_build
import torch
import contextlib
from tqdm import tqdm
import joblib 
import chemprop
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, median_absolute_error
from sklearn.decomposition import PCA
import json 
from utils import get_fingerprints, read_data, print_test_metrics, send_smiles_rt_to_csv
from sklearn.model_selection import cross_val_score, train_test_split, KFold
import os 
import numpy as np 

option = 'local'

os.chdir('10c51_project')
print("Current working directory: {0}".format(os.getcwd()))

option = 'local'

if option == 'online':
    filename = "https://figshare.com/ndownloader/files/18130628"
    smiles_list, retention_times = read_data(filename)
    fgp_list = get_fingerprints(smiles_list)
elif option == 'local':
    with open('fgp_list.txt', 'r') as f: fgp_list = json.loads(f.read())
    with open('smiles_list.txt', 'r') as f: smiles_list = json.loads(f.read())
    with open('retention_times.txt', 'r') as f: retention_times = json.loads(f.read())

train_csv, val_csv, test_csv, rt_test = send_smiles_rt_to_csv(smiles_list, retention_times)

train_loader, val_loader, _, _, _, fp_val, rt_val \
    = scale_split_build(fgp_list, retention_times)

folder_to_save = '/home/jfromer/10c51_project'

# train
path = os.path.join(folder_to_save, 'cp_ensembles_all_tmux.pth')
arguments = [
    '--data_path', 'DataFrame_train.csv',
    '--dataset_type', 'regression',
    '--save_dir', path,
    '--save_smiles_splits',
    '--batch_size', '512',
    '--gpu', '0', 
    '--epochs', '150', 
    '--ensemble_size', '5'
    ]
args = chemprop.args.TrainArgs().parse_args(arguments)
mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)
# show performance 


print('Ensembling compelete')