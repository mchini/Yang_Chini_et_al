# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:27:33 2020

@author: mchini
"""
#%% load packages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
from scipy.stats import zscore


path = 'your_folder_here'
path_figures = 'your_folder_here'
meta_data = pd.read_excel(path)[: 189]
animals = np.unique(meta_data['Mouse'])
sns.set()
sns.set_style("whitegrid")


#%% plot transients

conditions = ['awa', 'iso', 'fenta', 'keta']
columns4tsne = ['decay_no_isol', 'n_peaks', 'height', 'condition']
columns2log = ['n_peaks', 'height']
condition_rec = np.zeros(int(np.shape(meta_data)[0]))
idx_cond = 0

for idx_animal, animal in enumerate(animals):
    meta_animal = meta_data[meta_data['Mouse'] == animal]
    recordings = meta_animal['Number']
    transients4tSNEanimal = np.zeros((int(np.shape(recordings)[0]), 3))
    for idx_rec, idx in enumerate(np.flip(range(int(len(recordings))))):
        with open(r'E:/Calcium Imaging/results/same neuron stuff/dataframes high thr/'
                  + str(animal) + '/transients_df_quiet' + str(idx), 
                  'rb') as input_file:
            df = pickle.load(input_file)
        df = pd.DataFrame(df) # put into a pandas df
        # correct quiet as awake
        df['condition'].loc[df['condition'] == 'quiet'] = 'awa'
        # log height and n_peaks, take care of infs
        df['n_peaks'].loc[df['n_peaks'] == 0] = np.nan
        df[columns2log] = np.log10(df[columns2log])
        transients4tSNEanimal[idx_rec, :] = np.nanmedian(df[columns4tsne[0 : 3]], axis=0)
        condition_rec[idx_cond] = np.flatnonzero((conditions == np.unique(df['condition']))) # condition of recording
        idx_cond = idx_cond + 1
    transients4tSNEanimal = zscore(transients4tSNEanimal, axis=0)
    if idx_animal == 0:
        transients4tSNE = transients4tSNEanimal
    else:
        transients4tSNE = np.concatenate((transients4tSNE, transients4tSNEanimal), axis = 0)

# delete NAs
NANidx = np.argwhere(np.isnan(transients4tSNE))[:, 0]
transients4tSNE = np.delete(transients4tSNE, NANidx, 0)
condition_rec = np.delete(condition_rec, NANidx, 0)


############ plot t-sne of transients features ############

current_palette = np.array(sns.color_palette())
new_colors = sns.color_palette(current_palette[np.r_[0, 2, 1, 3], :])

from sklearn.manifold import TSNE
tsne = TSNE()


for kk in range(10):
    transients_emb = tsne.fit_transform(transients4tSNE)
    plt.figure(); plt.axis('off')
    for cond_idx in np.unique(condition_rec):    
        plt.scatter(transients_emb[condition_rec == cond_idx, 0],
                    transients_emb[condition_rec == cond_idx, 1],
                    c = new_colors[int(cond_idx)])
    plt.savefig(path_figures + str(kk) + '.png')
    plt.savefig(path_figures + str(kk) + '.eps')
    plt.close()
