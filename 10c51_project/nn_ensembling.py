from utils import read_data, get_fingerprints, scale_split_build
from utils import MLPfp, run_training, loss_curve, make_parity, print_metrics
import torch
import contextlib
import json
import os 
from tqdm import tqdm
import joblib 

option = 'local'

if option == 'online':
    filename = "https://figshare.com/ndownloader/files/18130628"
    smiles_list, retention_times = read_data(filename)
    fgp_list = get_fingerprints(smiles_list)
elif option == 'local':
    with open('fgp_list.txt', 'r') as f: fgp_list = json.loads(f.read())
    with open('smiles_list.txt', 'r') as f: smiles_list = json.loads(f.read())
    with open('retention_times.txt', 'r') as f: retention_times = json.loads(f.read())

print('data loaded')

study = joblib.load("hyperopt_study.pkl")


# hyperparameters optimized 
hidden_dim = [study.best_trial.params['Hidden_dim_1'], \
        study.best_trial.params['Hidden_dim_2'], \
        study.best_trial.params['Hidden_dim_3'], \
        study.best_trial.params['Hidden_dim_4']]
learning_rate = study.best_trial.params['Learning Rate']
wd = study.best_trial.params['Weight decay']
num_epochs = study.best_trial.params['Number of Epochs']; 

device = 'cuda:1'
num_ensembles = 10

train_loader, val_loader, _, _, _, fp_val, rt_val \
    = scale_split_build(fgp_list, retention_times)

folder_to_save = '/home/jfromer/10c51_project/nn_ensembles'

for i in tqdm(range(num_ensembles)):
    torch.manual_seed(i*12)
    model = MLPfp(hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, factor=0.5)
    val_loss_curve, train_loss_curve = run_training(model, train_loader, val_loader, optimizer, num_epochs, device)

    # print performance of this ensemble
    _, _, test_loader, fp_test, rt_test, _, _\
    = scale_split_build(fgp_list, retention_times)
    rt_test, rt_preds = make_parity(rt_test, fp_test, model, device, 'parity_original_hps.png')
    print_metrics(rt_test, rt_preds, rt_val, fp_val, device, model)

    # save model 
    path = os.path.join(folder_to_save, 'nn_ensemble_{}.pth'.format(i))
    torch.save(model.state_dict(), path)

print('Ensembling compelete')