'''Author: Priyanka Raghavan
   Description: linear regression baseline (no hyperparameter optimization) for randomly split data'''

path = "C:/Users/ChemeGrad2020/OneDrive/10.C51/Project"
import sys
sys.path.insert(1,path)
from data.data_processing import read_randomsplit_data
from figs.plot_parity import plot_parity_chemprop
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import r2_score

def lin_reg():
    smiles_train,smiles_test,fgp_train,fgp_test,rt_train,rt_test = read_randomsplit_data()
    linear_model = LinearRegression()
    linear_model.fit(fgp_train,rt_train)
    rt_pred = linear_model.predict(fgp_test)

    # remove non-retained molecules
    #nrt_idx = [i for i in range(len(rt_test)) if rt_test[i] < 5]
    #rt_test = [rt_test[i] for i in range(len(rt_test)) if i not in nrt_idx]
    #rt_pred = [rt_pred[i] for i in range(len(rt_pred)) if i not in nrt_idx]

    mae = mean_absolute_error(rt_test,rt_pred)
    medae = median_absolute_error(rt_test,rt_pred)
    mre = mean_absolute_percentage_error(rt_test,rt_pred)
    r2 = r2_score(rt_test,rt_pred)
    plot_parity_chemprop(rt_test,rt_pred,'linreg_parity.png')

    return mae,medae,mre,r2
