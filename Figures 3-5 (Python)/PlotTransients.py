# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:30:19 2020

@author: mchini
"""
#%% load packages

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle

path_excel = 'your_folder_here'
pathNeuronID = 'your_folder_here'
path2load = 'your_folder_here'
path4figsTransients = 'your_folder_here'
meta_data = pd.read_excel(path_excel)
animals = np.unique(meta_data['Mouse'])
sns.set()
sns.set_style("whitegrid")


#%% load transients stuff

df = []
df4tsne = []
df_same_neuron = []
conditions = ['awa', 'iso', 'fenta', 'keta']

for idx_animal, animal in enumerate(animals):
    meta_animal = meta_data[meta_data['Mouse'] == animal]
    recordings = meta_animal['Number']
    for idx_rec in range(np.shape(recordings)[0]):
        with open(path2load + str(int(animal)) + '/transients_df_quiet' + 
                  str(idx_rec), 'rb') as input_file:
            df1 = pickle.load(input_file)
        df1 = pd.DataFrame(df1)
        df1['mouse'] = animal
        # fix the unit of measure to ms
        df1['decay_isol_10'] = df1['decay_isol_10'] * 1000
        # set to NaN recordings for which extraction failed
        df1['decay_isol_10'].loc[df1['decay_isol_10'] > 2000] = np.nan
        if len(df) == 0:
            df = df1
        else:
            df = [df, df1]
            df = pd.concat(df)
       

# set quiet condition to awake
df['condition'][df['condition'] == 'quiet'] = 'awa'
# log-transform height and num_peaks, taking care of avoiding problems with 0s
df['n_peaks'][df['n_peaks'] == 0] = np.nan
df[['n_peaks', 'height']] = np.log10(df[['n_peaks', 'height']])


#%% plot transients stuff

current_palette = np.array(sns.color_palette())
colors_violin = sns.color_palette(current_palette[np.r_[0, 2, 1, 3], :])

plt.figure(); sns.violinplot(x = 'condition', y = 'decay_isol_10',
          data = df, order = conditions, palette = colors_violin)
title = 'Calcium transient decay constant all transients'
plt.ylabel('Time (ms)'); plt.title(title)
plt.ylim([0, 1000])
plt.savefig(path4figsTransients + title + '.eps')
plt.savefig(path4figsTransients + title + '.png')

plt.figure(); sns.violinplot(x = 'condition', y = 'n_peaks', cut = 0, 
          order = conditions, data = df[np.isfinite(df['n_peaks'])],
          palette = colors_violin)
title = 'N° of calcium peaks'
plt.ylabel('Log of calcium peaks / minute'); plt.title(title)
plt.savefig(path4figsTransients + title + '.eps')
plt.savefig(path4figsTransients + title + '.png')

plt.figure(); sns.violinplot(x = 'condition', y = 'height', cut = 0, 
          order = conditions, data = df[np.isfinite(df['height'])],
          palette = colors_violin)
title = 'Height of transients'
plt.ylabel('Log of arbitrary units'); plt.title(title)
plt.ylim([2.3, 3.4])
plt.savefig(path4figsTransients + title + '.eps')
plt.savefig(path4figsTransients + title + '.png')

