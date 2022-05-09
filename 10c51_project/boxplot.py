import matplotlib.pyplot as plt
import numpy as np 
import os 

os.chdir('10c51_project')

# from google sheets file 10.C51 Project Results 
# https://docs.google.com/spreadsheets/d/1UJpPswjejUI4-ZcnnnjIR-_Nxv2XTPKo9pW__xHDQqE/edit?usp=sharing
FFNN = [0.5325, 0.503, 0.5588333333, 0.486, 0.4741666667]
MPNN = [0.86180735, 0.80968857, 0.78495646, 0.8652668, 0.881021]

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5,5))
labels = ['FFNN', 'MPNN']

bplot = ax.boxplot([FFNN, MPNN], vert=True, patch_artist=True, labels=labels)
ax.set_title('Variation in Model Performance (FFNN and Vanilla MPNN)')
ax.set_ylabel('Median Absolute Error')

colors = ['pink', 'lightblue', 'lightgreen']

for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

for median in bplot['medians']:
    median.set_color('black')

plt.savefig('Boxplot.png')