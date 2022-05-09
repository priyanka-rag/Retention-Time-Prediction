import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from sklearn.metrics import r2_score
import os 
import chemprop
import pandas as pd
import torch
import joblib
from utils import MLPfp, get_fingerprints, print_test_metrics

include_non_retained = 0

def plot_parity_chemprop(y_true, y_pred, filename, y_pred_unc=None):
    axmin = min(min(y_true), min(y_pred)) - 0.1*(max(y_true)-min(y_true))
    axmax = max(max(y_true), max(y_pred)) + 0.1*(max(y_true)-min(y_true))
    
    r2 = r2_score(y_true,y_pred)
    
    plt.plot([axmin, axmax], [axmin, axmax], '--k')
    plt.errorbar(y_true, y_pred, yerr=y_pred_unc, linewidth=0, color =  (196/255,147/255,176/255), marker='o', markeredgecolor='w', alpha=1, elinewidth=1)
    
    plt.xlim((axmin, axmax))
    plt.ylim((axmin, axmax))
    
    ax = plt.gca()
    ax.set_aspect('equal')
    
    at = AnchoredText(
        f"R^2 = {r2:.2f}", prop=dict(size=10), frameon=True, loc='upper left')
    at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    ax.add_artist(at)
    
    plt.xlabel('True RT')
    plt.ylabel('Predicted RT')
    
    plt.savefig(filename)
    
    return
    



os.chdir('10c51_project')
# neural network 
study = joblib.load("hyperopt_study.pkl")


# hyperparameters optimized 
hidden_dim = [study.best_trial.params['Hidden_dim_1'], \
        study.best_trial.params['Hidden_dim_2'], \
        study.best_trial.params['Hidden_dim_3'], \
        study.best_trial.params['Hidden_dim_4']]


device = 'cuda:1'
path = os.path.join('nn_ensembles', 'nn_ensemble_{}.pth'.format(0))
model = MLPfp(hidden_dim).to(device)
model.load_state_dict(torch.load(path))
model.eval()

df = pd.read_csv('DataFrame_test.csv')
true_rt = df['Retention Time']
smi = df.SMILES.values
fgps_test = get_fingerprints(smi)
rt_preds_test = np.array(model(torch.Tensor(fgps_test).to(device)).cpu().detach().numpy())

if include_non_retained == 0:
    nrt_idx = [i for i in range(len(true_rt)) if true_rt[i] < 5]
    true_rt = [true_rt[i] for i in range(len(true_rt)) if i not in nrt_idx]
    rt_preds_test = [rt_preds_test[i] for i in range(len(rt_preds_test)) if i not in nrt_idx]

plot_parity_chemprop(true_rt, rt_preds_test, 'parity_nn_on_test.png', y_pred_unc=None)
MARE, MAE, MedAE = print_test_metrics(true_rt, rt_preds_test)


# chemprop
arguments = [
    '--test_path', 'DataFrame_test.csv',
    '--preds_path', 'test_preds_reg_chemprop_parity.csv',
    '--checkpoint_dir', 'test_checkpoints_reg'
]

args = chemprop.args.PredictArgs().parse_args(arguments)
preds = chemprop.train.make_predictions(args=args)

df = pd.read_csv('DataFrame_test.csv')
df['preds'] = [x[0] for x in preds]
true_rt = df['Retention Time']
preds = np.array(preds)

if include_non_retained == 0:
    nrt_idx = [i for i in range(len(true_rt)) if true_rt[i] < 5]
    true_rt = [true_rt[i] for i in range(len(true_rt)) if i not in nrt_idx]
    preds = [preds[i] for i in range(len(preds)) if i not in nrt_idx]

plot_parity_chemprop(true_rt, preds, 'parity_chemprop_on_test.png', y_pred_unc=None)
MARE, MAE, MedAE = print_test_metrics(true_rt, preds)


