import collections
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np 
from sklearn.model_selection import train_test_split
import pandas as pd 
from sklearn.model_selection import cross_val_score

from rdkit import RDLogger   
RDLogger.DisableLog('rdApp.*') # turn off RDKit warning message 

def get_smiles_from_inchikey(inchikey):
    mol = Chem.MolFromInchi(inchikey)
    try: 
      smiles = Chem.MolToSmiles(mol)
      smiles = Chem.CanonSmiles(smiles)
    except: print(inchikey)
    else: return smiles

def genFingerprints(smiles_list): # generates Morgan fingerprints given a list of smiles strings
    fp_list = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol,2)
        arr = np.array((0,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fp, arr)
        fp_list.append(arr.tolist())
    
    return fp_list

def init_data_processing(genFps=1):
    df = pd.read_csv("https://figshare.com/ndownloader/files/18130628", sep=";")
    retention_times = df['rt'].tolist()
    pubchem_ids = df['pubchem'].tolist()
    inchi_list = df['inchi'].tolist()
    duplicates = [item for item, count in collections.Counter(pubchem_ids).items() if count > 1]
    if len(duplicates) == 0:
      smiles_list = [get_smiles_from_inchikey(inchi) for inchi in inchi_list] # get all the smiles

      # some compounds can't be rendered in RDKit, so we remove these from the entire database
      idx_remove = [i for i in range(len(smiles_list)) if smiles_list[i] is None]
      retention_times = [retention_times[i]/60 for i in range(len(smiles_list)) if i not in idx_remove] #convert to minutes
      pubchem_ids = [pubchem_ids[i] for i in range(len(smiles_list)) if i not in idx_remove]
      inchi_list = [inchi_list[i] for i in range(len(smiles_list)) if i not in idx_remove]
      smiles_list = [smiles_list[i] for i in range(len(smiles_list)) if i not in idx_remove]

      # sweet, now we can generate fingerprints for the remaining 79957 valid molecules
      if genFps:
        fgp_list = genFingerprints(smiles_list)
        assert len(retention_times) == len(pubchem_ids) == len(inchi_list) == len(smiles_list) == len(fgp_list)
      else:
        fgp_list = [] 
        assert len(retention_times) == len(pubchem_ids) == len(inchi_list) == len(smiles_list)

      return inchi_list,pubchem_ids,smiles_list,fgp_list,retention_times

def split_data(inchi_list,smiles_list,retention_time_list,toSave): #generates holdout test set and sends to a csv file, also sends remaining test/validation set to csv file
    random_seed = 1234
    inchi_train,inchi_test,smiles_train,smiles_test,rt_train,rt_test = train_test_split(inchi_list,smiles_list,retention_time_list,test_size=0.1,random_state=random_seed)

    # put into dataframes
    data_train = {'InChI': inchi_train, 'SMILES': smiles_train, 'RT': rt_train}
    data_test = {'InChI': inchi_test, 'SMILES': smiles_test, 'RT': rt_test}
    df_train = pd.DataFrame(data_train)
    df_test = pd.DataFrame(data_test)

    if toSave: #if we want to send to csv file
      df_train.to_csv('/home/jfromer/10c51_project/train_valid_set.csv')
      df_test.to_csv('/home/jfromer/10c51_project/holdout_test_set.csv')

    return df_train,df_test

def read_randomsplit_data():
    df_train = pd.read_csv('train_valid_set.csv')
    df_test = pd.read_csv('holdout_test_set.csv')
    return df_train,df_test

def startup(option):
  if option == 'local':
    df_train, df_test = read_randomsplit_data()
  elif option == 'online':
    inchi_list,_,smiles_list,_,retention_times = init_data_processing(genFps=0)
    df_train,df_test = split_data(inchi_list,smiles_list,retention_times,toSave=0)
  return df_train, df_test

def list_from_df(df):
  smiles_list = df['SMILES'].tolist()
  retention_times = df['RT'].tolist()
  return smiles_list, retention_times

if __name__ == '__main__':
  inchi_list,_,smiles_list,_,retention_times = init_data_processing(genFps=0)
  df_train,df_test = split_data(inchi_list,smiles_list,retention_times,toSave=1)