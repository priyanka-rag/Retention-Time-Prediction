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
def err_vs_unc(unc, err, filename):
        
    plt.plot(unc, err, linewidth=0, color =  (196/255,147/255,176/255), marker='o', markeredgecolor='w', alpha=1, elinewidth=1)
    
    plt.xlabel('Uncertainty Estimate')
    plt.ylabel('Absolute error between mean prediction and true RT (min)')
    
    plt.savefig(filename)
    
    return

def make_histogram(x, filename):
    
    plt.hist(x, bins=20)     
    plt.xlabel('Uncertainty in RT (minutes)')
    plt.ylabel('Occurrences')
    plt.savefig(filename)
    
    return

os.chdir('10c51_project')

df = pd.read_csv('DataFrame_test.csv')
true_rt = df['Retention Time']
smi = df.SMILES.values
fgps_test = get_fingerprints(smi)

num_ensembles = 5
device = 'cuda:1'
study = joblib.load("hyperopt_study.pkl")
hidden_dim = [study.best_trial.params['Hidden_dim_1'], \
        study.best_trial.params['Hidden_dim_2'], \
        study.best_trial.params['Hidden_dim_3'], \
        study.best_trial.params['Hidden_dim_4']]

for i in range(num_ensembles):
    path = os.path.join('nn_ensembles', 'nn_ensemble_{}.pth'.format(i))
    model = MLPfp(hidden_dim).to(device)
    model.load_state_dict(torch.load(path))
    model.eval()
    rt_preds_test = np.array(model(torch.Tensor(fgps_test).to(device)).cpu().detach().numpy())
    label = 'pred_{}'.format(i)
    df[label] = rt_preds_test
    print(i)
    print_test_metrics(true_rt,rt_preds_test)


preds = [df['pred_{}'.format(i)].values for i in range(5)]
preds = np.array(preds).transpose()
stdev = np.std(preds,axis=1)
means = np.mean(preds,axis=1)
error = np.abs(np.subtract(means, np.array(true_rt).ravel()))

make_histogram(stdev, 'stdev_nn.png')

print('Prediction Means')
print_test_metrics(true_rt,means)
