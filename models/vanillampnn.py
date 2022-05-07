'''Author: Priyanka Raghavan
   Description: vanilla mpnn baseline (no edge features) for randomly split data
   Note: this is adapted from my answers to the 10.C51 Problem Set 4, the skeleton code of which was 
         created by the 10.C51 course staff'''

path = "C:/Users/ChemeGrad2020/OneDrive/10.C51/Project"
import sys
sys.path.insert(1,"C:/Users/ChemeGrad2020/OneDrive/10.C51/Project")
from data.data_processing import read_randomsplit_data
from figs.plot_parity import plot_parity_chemprop
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import collections
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors
import sys
import torch 
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset,DataLoader
from torch import nn
from torch.nn import ModuleDict
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error
from torch import optim
import random
from sklearn.model_selection import cross_val_score

from rdkit import RDLogger   
RDLogger.DisableLog('rdApp.*') # turn off RDKit warning message

def smiles2graph(smiles):
    '''
    Transform smiles into a list of atomic numbers and an edge array
    
    Args: 
        smiles (str): SMILES strings
    
    Returns: 
        z(np.array), A (np.array): list of atomic numbers, edge array
    '''
    
    mol = Chem.MolFromSmiles( smiles ) # no hydrogen 
    z = np.array( [atom.GetAtomicNum() for atom in mol.GetAtoms()] )
    A = np.stack(Chem.GetAdjacencyMatrix(mol)).nonzero()
    
    return z, A

def featurize(smiles_list,retention_times):
  AtomicNum_list = []
  Edge_list = []
  y_list = []
  y = retention_times
  Natom_list = []

  for idx,smi in enumerate(smiles_list):
    z,A = smiles2graph(smi)
    AtomicNum_list.append(torch.LongTensor(z))
    Edge_list.append(torch.LongTensor(A))
    Natom_list.append(int(len(z)))
    y_list.append(torch.FloatTensor([y[idx]]))
  
  return AtomicNum_list,Edge_list,Natom_list,y_list

class GraphDataset(torch.utils.data.Dataset):
    def __init__(self,
                 AtomicNum_list, 
                 Edge_list, 
                 Natom_list, 
                 y_list):
        
        '''
        GraphDataset object
        
        Args: 
            z_list (list of torch.LongTensor)
            a_list (list of torch.LongTensor)
            N_list (list of int)
            y_list (list of torch.FloatTensor)

        '''
        self.AtomicNum_list = AtomicNum_list # atomic number
        self.Edge_list = Edge_list # edge list 
        self.Natom_list = Natom_list # Number of atoms 
        self.y_list = y_list # properties to predict 

    def __len__(self):
        return len(self.Natom_list)

    def __getitem__(self, idx):
        
        AtomicNum = torch.LongTensor(self.AtomicNum_list[idx])
        Edge = torch.LongTensor(self.Edge_list[idx])
        Natom = self.Natom_list[idx]
        y = torch.Tensor(self.y_list[idx])
        
        return AtomicNum, Edge, Natom, y

def collate_graphs(batch):
    '''Batch multiple graphs into one batched graph
    
    Args:
    
        batch (tuple): tuples of AtomicNum, Edge, Natom and y obtained from GraphDataset.__getitem__() 
        
    Return 
        (tuple): Batched AtomicNum, Edge, Natom, y
    
    '''
    
    AtomicNum_batch = []
    Edge_batch = []
    Natom_batch = []
    y_batch = []

    cumulative_atoms = np.cumsum([0] + [b[2] for b in batch])[:-1]
    
    for i in range(len(batch)):
        z, a, N, y = batch[i]
        index_shift = cumulative_atoms[i]
        a = a + index_shift
        AtomicNum_batch.append(z) 
        Edge_batch.append(a)
        Natom_batch.append(N)
        y_batch.append(y)
        
    AtomicNum_batch = torch.cat(AtomicNum_batch)
    Edge_batch = torch.cat(Edge_batch, dim=1)
    Natom_batch = Natom_batch
    y_batch = torch.cat(y_batch)
    
    return AtomicNum_batch, Edge_batch, Natom_batch, y_batch 

def scatter_add(src, index, dim_size, dim=-1, fill_value=0):
    
    '''
    Sums all values from the src tensor into out at the indices specified in the index 
    tensor along a given axis dim. 
    '''
    
    index_size = list(itertools.repeat(1, src.dim()))
    index_size[dim] = src.size(dim)
    index = index.view(index_size).expand_as(src)
    
    dim = range(src.dim())[dim]
    out_size = list(src.size())
    out_size[dim] = dim_size

    out = src.new_full(out_size, fill_value)

    return out.scatter_add_(dim, index, src)

class VanillaMPNN(torch.nn.Module):
    '''
        A Vanilla MPNN model 
    '''
    def __init__(self, n_convs=6, n_embed=48):
        super(VanillaMPNN, self).__init__()
        
        self.atom_embed = nn.Embedding(100, n_embed)
        # Declare MLPs in a ModuleList
        self.convolutions = nn.ModuleList(
            [ 
                ModuleDict({
                    'update_mlp': nn.Sequential(nn.Linear(n_embed, n_embed), 
                                                nn.ReLU(), 
                                                nn.Linear(n_embed, n_embed)),
                    'message_mlp': nn.Sequential(nn.Linear(n_embed, n_embed), 
                                                 nn.ReLU(), 
                                                 nn.Linear(n_embed, n_embed)) 
                })
                for _ in range(n_convs)
            ]
            )
        # Declare readout layers
        self.readout = nn.Sequential(nn.Linear(n_embed, n_embed), nn.ReLU(), nn.Linear(n_embed, 1))
        
    def forward(self, AtomicNum, Edge, Natom):

        # Parametrize embedding 
        h_t = self.atom_embed(AtomicNum) #eqn. 1

        for layer in self.convolutions:
          update_layer = layer['update_mlp']
          message_layer = layer['message_mlp']

          # Message step
          h_i = torch.index_select(h_t,0,Edge[0])
          h_j = torch.index_select(h_t,0,Edge[1])
          msg = message_layer(torch.mul(h_i,h_j))
          #m_t = scatter_add(src=msg, index=Edge[1], dim=0, dim_size=len(AtomicNum)) +  scatter_add(src=msg, index=Edge[0], dim=0, dim_size=len(AtomicNum))
          m_t = scatter_add(src=msg, index=Edge[1], dim=0, dim_size=len(AtomicNum))

          # Update step
          h_t = torch.add(h_t, update_layer(m_t))

        # Readout step
        y_readout = torch.split(self.readout(h_t), Natom) # readout and split tensor back into original graph sizes
        output = torch.stack([torch.sum(graph) for graph in y_readout]) # sum all final node embeddings within each graph

        return output

def loop(model, optimizer, device, loader, epoch, evaluation=False):
    
    if evaluation:
        model.eval()
        mode = "eval"
    else:
        model.train()
        mode = 'train'
    batch_losses = []
    
    # Define tqdm progress bar 
    tqdm_data = tqdm(loader, position=0, leave=True, desc='{} (epoch #{})'.format(mode, epoch))
    
    for data in tqdm_data:
        
        AtomicNumber, Edge, Natom, y = data 
        AtomicNumber = AtomicNumber.to(device)
        Edge = Edge.to(device)
        y = y.to(device)
        pred = model(AtomicNumber, Edge, Natom)
        loss = (pred-y).pow(2).mean() # MSE loss
        
        if not evaluation:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        batch_losses.append(loss.item())

        postfix = ['batch loss={:.3f}'.format(loss.item()) , 
                   'avg. loss={:.3f}'.format(np.array(batch_losses).mean())]
        
        tqdm_data.set_postfix_str(' '.join(postfix))
    
    return np.array(batch_losses).mean()

def save_model(state,fpath):
    torch.save(state,fpath)
    return

def load_model(fpath, model, optimizer):
    checkpoint = torch.load(fpath)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    train_losses = checkpoint['train losses']
    val_losses = checkpoint['val losses']
    return model, optimizer, train_losses, val_losses

def train_model(smiles_train,smiles_test,rt_train,rt_test,num_epochs,name):
    # first featurize randomly split data
    AtomicNum_tr,Edge_tr,Natom_tr,y_tr = featurize(smiles_train,rt_train)
    AtomicNum_ts,Edge_ts,Natom_ts,y_ts = featurize(smiles_test,rt_test)
    AtomicNum_tr, AtomicNum_val, Edge_tr, Edge_val, Natom_tr, Natom_val, y_tr, y_val = train_test_split(AtomicNum_tr,Edge_tr,Natom_tr,y_tr, test_size = 0.1/0.8)

    # generate datasets and dataloaders
    train_dataset = GraphDataset(AtomicNum_tr,Edge_tr,Natom_tr,y_tr)
    val_dataset = GraphDataset(AtomicNum_val,Edge_val,Natom_val,y_val)
    test_dataset = GraphDataset(AtomicNum_ts,Edge_ts,Natom_ts,y_ts)

    train_loader = DataLoader(train_dataset,
                            batch_size=512, 
                            collate_fn=collate_graphs,shuffle=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=512, 
                            collate_fn=collate_graphs,shuffle=True)

    test_loader = DataLoader(test_dataset,
                            batch_size=512, 
                            collate_fn=collate_graphs,shuffle=True)
    
    #define model and optimizer 
    device = 'cuda:0'
    model = VanillaMPNN(n_convs=6, n_embed=48).to(device) # hyperparams from the paper
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, verbose=True)

    # now train and save the model
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):    
        train_loss = loop(model, optimizer, device, train_loader, epoch)
        val_loss = loop(model, optimizer, device, val_loader, epoch, evaluation=True)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        scheduler.step(val_loss)
    
    final_model = {'state_dict': model.state_dict(),
                    'optimizer':  optimizer.state_dict(),
                    'train losses': train_losses,
                    'val losses': val_losses}
    save_model(final_model, path + "/saved_models/" + name + ".pt")
    return model

def plot_loss_curve(train_losses,val_losses):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(train_losses, label='Training Loss')
    ax.plot(val_losses, label='Validation Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend(loc='upper right')
    fig.tight_layout()
    plt.savefig(path + '/figs/vanillampnn_loss_curve.png')
    return 

def predict_retention_times(model,loader,device):
  y_true = []
  y_pred = []

  model.eval()
  for batch in loader:
        AtomicNumber, Edge, Natom, y = batch 
        AtomicNumber = AtomicNumber.to(device)
        Edge = Edge.to(device)
        y = y.to(device)
        pred = model(AtomicNumber, Edge, Natom)
        y_true.append(y.cpu().detach().numpy())
        y_pred.append(pred.cpu().detach().numpy())
  
  y_true = np.concatenate(y_true).ravel()
  y_pred = np.concatenate(y_pred).ravel()
  mae = mean_absolute_error(y_true,y_pred)
  medae = median_absolute_error(y_true,y_pred)
  mre = mean_absolute_percentage_error(y_true,y_pred)
  r2 = r2_score(y_true,y_pred)

  return y_true,y_pred,mae,medae,mre,r2
