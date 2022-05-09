from utils import optimize_parameters, get_fingerprints, split_sets
import os 
from data_processing import startup, list_from_df


os.chdir('10c51_project')
option = 'local'
df_train_val, df_test = startup(option)
smi_train_val, rt_train_val = list_from_df(df_train_val)
print('generating training and validation fingerprints')
fp_train_val = get_fingerprints(smi_train_val)
train_data, val_data = split_sets(fp_train_val, rt_train_val)

device = 'cuda:1'

n_trials = 200
study = optimize_parameters(train_data, val_data, device, n_trials)

print('Run complete, with optimal parameters:')

for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")