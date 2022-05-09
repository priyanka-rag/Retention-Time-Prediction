import chemprop
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, median_absolute_error
from sklearn.decomposition import PCA
import json 
from utils import get_fingerprints, read_data, send_smiles_rt_to_csv, plot_parity_chemprop, unscale 
from sklearn.model_selection import cross_val_score, train_test_split, KFold
import os 
import numpy as np 

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

# train 
arguments = [
    '--data_path', 'DataFrame_train.csv',
    '--dataset_type', 'regression',
    '--checkpoint_dir','test_checkpoints_reg',
    '--save_dir', 'test_checkpoints_reg',
    '--save_smiles_splits',
    '--batch_size', '512',
    '--gpu', '1', 
    '--epochs', '100'
]

args = chemprop.args.TrainArgs().parse_args(arguments)
mean_score, std_score = chemprop.train.cross_validate(args=args, train_func=chemprop.train.run_training)

# predict 
arguments = [
    '--test_path', 'DataFrame_test.csv',
    '--preds_path', 'test_preds_reg.csv',
    '--checkpoint_dir', 'test_checkpoints_reg'
]

args = chemprop.args.PredictArgs().parse_args(arguments)
rt_preds = np.array(chemprop.train.make_predictions(args=args))
rt_test = unscale(rt_test)
rt_preds = unscale(rt_preds)

# parity plot and performance 
print('TEST SET METRICS')
MARE = mean_absolute_percentage_error(rt_test, rt_preds)
print('Mean Relative Error: ' + str(MARE))
MAE = mean_absolute_error(rt_test, rt_preds)
print('Mean Absolute Error: ' + str(MAE))
MedAE = median_absolute_error(rt_test, rt_preds)
print('Median Absolute Error: ' + str(MedAE))

filename = 'parity_chemprop.png'
plot_parity_chemprop(rt_test, rt_preds, filename, y_pred_unc=None)
