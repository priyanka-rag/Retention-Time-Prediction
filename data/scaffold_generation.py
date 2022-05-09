# -*- coding: utf-8 -*-
"""
Created on Sun May  8 13:33:38 2022

@author: Nathan Stover
"""

import numpy as np
import pandas as pd 
import requests
import matplotlib.pyplot as plt
import collections
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPRegressor
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from hyperopt import hp
from hyperopt import fmin, tpe, space_eval, Trials
from sklearn.cluster import KMeans
import matplotlib.cm as cm
import pandas as pd
import random as rand

#Use initial data processing to generate lists of smiles and fingerprints
#Functions in this file generate data splits 


#These next three functions visualize data splits for molecules. To fun these functions, 

# import pandas as pd
# class_clusters = pd.read_csv ('/content/drive/MyDrive/10.C51 Project/scaffold_split.csv')
# id_array = class_clusters.to_numpy().astype(int)
# #Change split_n to 1, 2, or 3 to investigate different splits
# split_n = 3
# juxtapose(split_n,id_array)



def gen_random_ind(split_type,split_num, array):
  #Split type 1, 2, 3
  #Split num 0:num_splits-1
  while True:
    site_id = rand.choice(range(np.shape(array)[0]))
    if array[site_id,split_type] == split_num:
      return site_id

def get_strucs(split_type,split_num, array, num_mols):
  ls = []
  for i in range(num_mols):
    ind = gen_random_ind(split_type,split_num, array)
    #print(smiles_list[ind])
    struc = Chem.MolFromSmiles(smiles_list[ind])
    ls.append(struc)
  return ls


def juxtapose(split_type,array):
  for i in range(5):
    struc_ls = get_strucs(split_type,i, array, 5)
    print("Cluster "+str(i))
    fig = Draw.MolsToGridImage(struc_ls)
    display(fig)
    
    
#These next functions perform k-means clustering on PCA of fingerprints 
#Three options on how to use these sets of functions:
#1:  get_PCA_clustered_groups(num_fgp_pca_clust,fgp_list) generates array of cluster labels
#In same form as fgp_list, where first are is a the num of clusters to use

#For 2 and 3, run
  # pca = PCA(n_components=400)
  # pca.fit(fgp_list)
  # fgp_pca = pca.transform(fgp_list)
  
#2 elbow(pca_dimensions,max_clusters) to generate elbow plot, useful for picking clustering params

#3 cluster_plot(num_dim,num_clust) to generate a 2D plot that shows distribution of clusters. 

def get_PCA_clustered_groups(num_pca_cluster_groups,fgp_list):
  pca = PCA(n_components=400)
  pca.fit(fgp_list)
  fgp_pca = pca.transform(fgp_list)
  clust = cluster_eval_grouper(10,num_pca_cluster_groups,fgp_pca)
  return clust.labels_
  
def cluster_eval_grouper(num_pca_dims,num_clusters,fgp_pca):
  cluster = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto')
  cluster.fit(fgp_pca[:,:num_pca_dims])
  return cluster

def cluster_eval(num_pca_dims,num_clusters):
    #k-means clustering on PCA data
  cluster = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto')
  cluster.fit(fgp_pca[:,:num_pca_dims])
  return cluster

def cluster_plot(num_pca_dim,num_clusters):
  #Plotting to show PCA clustering, play around with parameters
  cluster = cluster_eval(num_pca_dim,num_clusters)
  colors = cm.Dark2(range(num_clusters))
  fig, ax = plt.subplots(figsize=(10,10))
  ax.scatter(fgp_pca[:,0], fgp_pca[:,1], s=3,c=colors[cluster.labels_]) 

def elbow(num_pca_dims,max_clust):
  #Create Elbow plot
  elbow_inertia = np.zeros(max_clust)
  for clust_num in range(max_clust):
    cluster = cluster_eval(num_pca_dims,clust_num+1)
    elbow_inertia[clust_num] = cluster.inertia_
  plt.plot(range(1,max_clust+1),elbow_inertia)
  plt.xlabel("Number of Clusters")
  plt.ylabel("Internal Distance")
  
  
#These next functions perform scaffold splitting with rdkit
#This function takes a list of smiles and a number of groups to generate, and uses scaffold splitting to assign each molecule to a group.
#Returns a np array containing the raw group id and super-group id of each molecule. 
#This is random clustering of 
#example: scaffold_array,cluster_list = raw_scaffold(5,smiles_list)
def raw_scaffold(num_groups,smile_ls):
  num_molecules = len(smile_ls)
  cluster_list = []
  num_scaffolds = 1
  scaffold_splits = []
  for i in range(num_molecules):
    # define compound via its SMILES string
    smiles = smile_ls[i]
    # convert SMILES string to RDKit mol object 
    mol = Chem.MolFromSmiles(smiles)
    # create RDKit mol object corresponding to Bemis-Murcko scaffold of original compound
    mol_scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    # make the scaffold generic by replacing all atoms with carbons and all bonds with single bonds
    mol_scaffold_generic = MurckoScaffold.MakeScaffoldGeneric(mol_scaffold)
    # convert the generic scaffold mol object back to a SMILES string format
    smiles_scaffold_generic = Chem.CanonSmiles(Chem.MolToSmiles(mol_scaffold_generic))
    #determine scaffold number
    cluster_num = None
    for cluster_ind,candidate_scaffold in enumerate(cluster_list):
      if smiles_scaffold_generic == candidate_scaffold:
        cluster_num = cluster_ind
    if cluster_num == None:
      cluster_num = num_scaffolds
      num_scaffolds+=1
      cluster_list.append(smiles_scaffold_generic)
    scaffold_splits.append(cluster_num)
 
  scaffold_splits_grouped = np.zeros(num_molecules)
  cutoff_array = np.ceil(max(scaffold_splits)*np.linspace(0,1,num_groups+1))
  for i in range(num_molecules):
    scaffold_splits_grouped[i] = np.sum(np.less(cutoff_array,np.array(scaffold_splits)[i]))-1
  scaff_array = np.zeros((num_molecules,2))
  scaff_array[:,0] = np.array(scaffold_splits)
  scaff_array[:,1] = scaffold_splits_grouped
  return (scaff_array,cluster_list)

#These next functions perform k_means clustering based on the scaffolds given in the function above. 
#Returns np array of cluster nums following smiles_list order
#example call: 
#scaffold_array,cluster_list = raw_scaffold(num_scaffold_groups,smiles_list)
#scaffold_kmeans = gen_scaff_cluster_array(num_spca_cluster,cluster_list)

#k-means clustering on PCA data
def sc_cluster_eval(num_pca_dims,num_clusters,scaff_fgp_pca):
  cluster = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10, max_iter=300, tol=0.0001, verbose=0, random_state=None, copy_x=True, algorithm='auto')
  cluster.fit(scaff_fgp_pca[:,:num_pca_dims])
  return cluster

def get_pca_scaffold(fg_ls):
  scaffold_fgp_list = fg_ls
  pca = PCA(n_components=400)
  pca.fit(scaffold_fgp_list)
  scaffold_fgp_pca = pca.transform(scaffold_fgp_list)
  return (scaffold_fgp_pca,pca)

def gen_scaff_cluster_array(num_spca_cluster,fg_ls):
  num_pca_dims = 10
  scaff_fgp_pca,pca_u = get_pca_scaffold(fg_ls)
  clust = sc_cluster_eval(num_pca_dims,num_spca_cluster,scaff_fgp_pca)
  return clust.labels_