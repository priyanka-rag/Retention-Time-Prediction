'''Author: Priyanka Raghavan
   Description: vanilla mpnn ensembling to estimate epistemic uncertainty'''

path = "C:/Users/ChemeGrad2020/OneDrive/10.C51/Project"
import sys
sys.path.insert(1,"C:/Users/ChemeGrad2020/OneDrive/10.C51/Project")
import numpy as np
import matplotlib.pyplot as plt
from data.data_processing import read_randomsplit_data
from figs.plot_parity import plot_parity_chemprop
from vanillampnn import GraphDataset,featurize,collate_graphs,train_model,predict_retention_times
import torch
from torch.utils.data import TensorDataset,DataLoader
from torch import nn

def mpnn_ensembling():
    smiles_train,smiles_test,fgp_train,fgp_test,rt_train,rt_test = read_randomsplit_data()
    AtomicNum_ts,Edge_ts,Natom_ts,y_ts = featurize(smiles_test,rt_test)
    test_dataset = GraphDataset(AtomicNum_ts,Edge_ts,Natom_ts,y_ts)
    test_loader = DataLoader(test_dataset,
                            batch_size=512, 
                            collate_fn=collate_graphs,shuffle=True)

    # train 5 randomly-initialized models and record predictions on the test set
    predictions = []

    for i in range(5):
        ensemble_model = train_model(smiles_train,smiles_test,rt_train,rt_test,750,'vanillampnn_rt_model_ensemble_' + str(i),True)
        y_true,y_pred,mae,medae,mpe,r2 = predict_retention_times(ensemble_model,test_loader)

        # remove non-retained molecules
        nrt_idx = [i for i in range(len(y_true)) if y_true[i] < 5]
        y_pred = [y_pred[i] for i in range(len(y_pred)) if i not in nrt_idx]

        predictions.append(y_pred)
    
    avgs = [sum(col)/float(len(col)) for col in zip(*predictions)]
    stds = [np.std(col) for col in zip(*predictions)]
    plt.scatter(avgs,stds,color = (196/255,147/255,176/255), marker='o', edgecolor='w', alpha=1)
    plt.xlabel('Average Predicted RT (min)')
    plt.ylabel('Uncertainty (min)')
    plt.savefig(path + '/figs/vanillampnn_ensembling_results.png')

    return avgs,stds
