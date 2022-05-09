from utils import get_fingerprints, split_build_train
from utils import MLPfp, run_training, loss_curve, make_parity, print_metrics
import torch
from data_processing import startup, list_from_df
import json
import os 

os.chdir('10c51_project')
option = 'local'
df_train_val, df_test = startup(option)
smi_train_val, rt_train_val = list_from_df(df_train_val)
fp_train_val = get_fingerprints(smi_train_val)

train_loader, val_loader, fp_val, rt_val= split_build_train(fp_train_val, rt_train_val)

device = 'cuda:1'
hidden_dim = [1000, 500, 200, 100]
learning_rate = 0.001
wd = 0.0001; 
num_epochs = 20; 

torch.manual_seed(2)
model = MLPfp(hidden_dim).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=wd)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True, factor=0.5)

val_loss_curve, train_loss_curve = run_training(model, train_loader, val_loader, optimizer, num_epochs, device)


# Evaluate performance 
smi_test, rt_test = list_from_df(df_test)
fp_test = get_fingerprints(smi_test)
rt_test, rt_preds = make_parity(rt_test, fp_test, model, device, 'parity_original_hps.png')

print_metrics(rt_test, rt_preds, rt_val, fp_val, device, model)

loss_curve(val_loss_curve, train_loss_curve, 'loss_curve_orig_hps.png')

print('run complete')