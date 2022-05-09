# -*- coding: utf-8 -*-
"""
Created on Sun May  8 13:43:59 2022

@author: Nathan Stover
"""
from rdkit.Chem import AllChem
from rdkit import DataStructs
from rdkit.ML.Cluster import Butina

#These two functions are used to generate tanimoto distance list between test and training sets. 
def genFingerprints_bitvec(smiles_list): # generates Morgan fingerprints in rdk friendly bit vector form given a list of smiles strings
    fp_list = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol,2)
        fp_list.append(fp)
    return fp_list

def bulk_tan_dis(train_ls, unknown_ls,k):
  #Takes two lists of smiles strings, and a number of nearest neighbors
  #Returns a list of avg distances corresponding to unknown_ls
  train_btv = genFingerprints_bitvec(train_ls)
  test_btv = genFingerprints_bitvec(unknown_ls)
  uncert_list = []
  for i in test_btv:
          sims = DataStructs.BulkTanimotoSimilarity(i,train_btv)
          sims.sort()
          sims.reverse()
          uncert_list.append(np.mean([-np.log2(j) for j in sims[:k]]))
  return uncert_list

#This function used to plot tanimoto distance from modified MPNN.
def plot_tan_dist(model,test_loader):
  y_error,tan_ls = predict_retention_times(model,test_loader)
  b, a = np.polyfit(tan_ls, y_error, deg=1)

  # Create sequence of 100 numbers from 0 to 100 
  xseq = np.linspace(0, 3, num=100)

  # Plot regression line
 # ax.plot(xseq, a + b * xseq, color="k", lw=2.5);

  fig, ax = plt.subplots(figsize=(6, 6))
  sns.scatterplot(
      x=tan_ls,
      y=y_error,
      color="k",
      ax=ax,
      marker="."
  )
  sns.kdeplot(
      x=tan_ls,
      y=y_error,
      levels=5,
      fill=True,
      alpha=0.6,
      cut=2,
      ax=ax,
  )
  ax.plot(xseq, a + b * xseq, color="k", lw=2.5)

  ax.set_ylabel('Absolute Error')
  ax.set_xlabel('Tanimoto Distance')
  
  #This func used to run model loading, analysis,plotting. Used with Vanilla MPNN
def run_tan_plotting():
    #Load split data and model
    device = 'cuda:0'
    fgp_split_path= "/content/vanillampnn_rt_model_fgp_split.pt"
    model = VanillaMPNN().to(device)
    model.load_state_dict(torch.load(fgp_split_path))
    model.eval()
    
    splitting_df = pd.read_csv ('scaffold_split.csv')
    #Split between test and train sets
    in_train_ls = (splitting_df["Kmeans from PCA"]==3.0).tolist()
    fgp_train_ls = [smiles_list[ind] for ind,val in enumerate((splitting_df["Kmeans from PCA"]==3.0).tolist()) if val]
    fgp_test_ls = [smiles_list[ind] for ind,val in enumerate((splitting_df["Kmeans from PCA"]==3.0).tolist()) if not(val)]
    
    fgp_train_df = splitting_df[splitting_df["Kmeans from PCA"]==3.0]
    fgp_eval_df = splitting_df[splitting_df["Kmeans from PCA"]!=3.0]
    fgp_tan_dist = bulk_tan_dis(fgp_train_ls,fgp_test_ls,8)
    
    AtomicNum_list = []
    Edge_list = []
    y_list = []
    y = retention_times
    Natom_list = []
    #Add non-train data to the dataloaders
    for idx,smi in enumerate(smiles_list):
      if not(in_train_ls[idx]):
        z,A = smiles2graph(smi)
        AtomicNum_list.append(torch.LongTensor(z))
        Edge_list.append(torch.LongTensor(A))
        Natom_list.append(int(len(z)))
        y_list.append(torch.FloatTensor([y[idx]/60]))
    
    
    test_dataset = GraphDataset(AtomicNum_list,Edge_list,Natom_list,y_list,fgp_tan_dist)
    
    test_loader = DataLoader(test_dataset,
                              batch_size=512, 
                              collate_fn=collate_graphs,shuffle=True)
    #Plot with model
    plot_tan_dist(model,test_loader)