# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:25:10 2020

@author: mchini
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle


path_excel = 'your_path_here'
path2load = 'your_path_here'
meta_data = pd.read_excel(path_excel)
animals = np.unique(meta_data['Mouse'])
sns.set()
sns.set_style("whitegrid")
df = []
distance_bins = np.linspace(0, 500, 11) # bins for xcorr over distance plot
distance_microns = distance_bins * 0.41 # convert to micrometers


#%% load and plot pairwise corr stuff only for traces

for animal in animals:
    meta_animal = meta_data[meta_data['Mouse'] == animal]
    recordings = meta_animal['Number']
    for idx, recording in enumerate(np.flip(recordings)):
        with open(path2load + str(animal) + '/corr_df_quiet_z' 
                  + str(idx), 'rb') as input_file:
            df1 = pickle.load(input_file)
            df1['mouse'] = animal # set animal (for animal by animal statistics)
            df1['recording'] = recording # set animal (for animal nested stats)
            df1['CorrSpikesSpearman'] = df1['CorrSpikesSpearman'].flatten()
            df1['CorrTracessSpearman'] = df1['CorrTracessSpearman'].flatten()
            if len(df) == 0:
                df = pd.DataFrame(df1)
            else:
                df = [df, pd.DataFrame(df1)]
                df = pd.concat(df)

# reassign animals 148-151-153 
df['mouse'][df['mouse'] == 148] = 48
df['mouse'][df['mouse'] == 151] = 51
df['mouse'][df['mouse'] == 153] = 53
df['condition'][df['condition'] == 'quiet'] = 'awa'
conditions = np.unique(df['condition']) # put only here because you adjusted quiet
animals = np.unique(df['mouse']) # put only here because you adjusted animals 148, 151 and 153

# correct bias with Fisher z-transformation
df['TcorrFisher'] = np.tanh(df['CorrTraces'])
df['AbsTcorrFisher'] = np.abs(df['TcorrFisher'])
df['TcorrSpearmanFisher'] = np.tanh(df['CorrTracessSpearman'])

#%% prepare and save stuff for R

####### this is to extract by-recording stats (raw numbers are too big) #######
cols4grouping = ['mouse', 'recording', 'condition']
df_by_rec = df.groupby(cols4grouping).mean()
df_by_rec['mouse'] = df_by_rec.index.get_level_values(0)
df_by_rec['condition'] = df_by_rec.index.get_level_values(2)
df_by_rec['recording'] = df_by_rec.index.get_level_values(1)
df_by_rec = df_by_rec.reset_index(drop=True)

# this part is to compare proportion of values in 1st and 4th quartile for Tcorr
# initialize  stuff
prop_qrt1 = np.zeros((np.shape(animals)[0],
                      np.shape(conditions)[0]))
prop_qrt4 = np.zeros(np.shape(prop_qrt1))

# loop through mice and condition, compute proportion
for idx_mouse, mouse in enumerate(animals):
    df_animal = df[df['mouse'] == mouse]
    for idx_cond, condition in enumerate(conditions):
        df_cond = df_animal['TcorrFisher'][df_animal['condition'] == condition]
        # compute 1st and 4th quartile for each mouse
        qrt1 = np.quantile(df_animal['TcorrFisher'], 0.25)
        qrt4 = np.quantile(df_animal['TcorrFisher'], 0.75)
        # check that df is not empty
        if len(df_cond) > 0:
            prop_qrt1[idx_mouse, idx_cond] = np.count_nonzero(df_cond < qrt1) / len(df_cond)
            prop_qrt4[idx_mouse, idx_cond] = np.count_nonzero(df_cond > qrt4) / len(df_cond)
        else:
            prop_qrt1[idx_mouse, idx_cond] = np.nan
            prop_qrt4[idx_mouse, idx_cond] = np.nan

df_by_rec.to_excel('E:/Calcium Imaging/stats/datasets/correlation/high thr by rec.xlsx')

#%% plot stuff

###### plot cdf for Pearson correlation ######
# set bins in which to compute cdf
bins = np.linspace(-0.5, 0.5, 500)
# plot cdf for every condition
fig, ax = plt.subplots()
for condition in conditions:
    to_plot = df['TcorrFisher'][df['condition'] == condition]
    n, bins, patches = ax.hist(to_plot,
                               bins, density = True,
                               histtype = 'step', 
                               cumulative = True,
                               label = condition)
# fix axis and labels
plt.xlim(-0.3, 0.5); ax.legend(loc = 4); 
plt.ylabel('Cumulative density function'); plt.xlabel('Correlation coeff (z)');
plt.title('Fisher-transformed correlation - Traces')

###### plot cdf for Spearman correlation ######
# set bins in which to compute cdf
bins = np.linspace(-0.5, 0.5, 500)
# plot cdf for every condition
fig, ax = plt.subplots()
for condition in conditions:
    to_plot = df['CorrTracessSpearman'][df['condition'] == condition]
    n, bins, patches = ax.hist(to_plot,
                               bins, density = True,
                               histtype = 'step', 
                               cumulative = True,
                               label = condition)
# fix axis and labels
plt.xlim(-0.3, 0.5); ax.legend(loc = 4); 
plt.ylabel('Cumulative density function'); plt.xlabel('Spearman correlation coeff (z)');
plt.title('Fisher-transformed Spearman correlation - Traces')

# put quartile stuff into pandas dataframe, assign columns and plot
prop_qrt1 = pd.DataFrame(prop_qrt1)
prop_qrt1.columns = conditions
plt.figure()
sns.swarmplot(data = prop_qrt1)
plt.ylabel('Proportion in 1st quartile'); plt.xlabel('Condition');
        
prop_qrt4 = pd.DataFrame(prop_qrt4)
prop_qrt4.columns = conditions
plt.figure()
sns.swarmplot(data = prop_qrt4)
plt.ylabel('Proportion in 4th quartile'); plt.xlabel('Condition');
   
# subdivide distance into 10 bins and name these bins
df['distance bin'] = pd.cut(df['distance'], distance_bins,
  labels=["0-20", "20-40", "40-60", "60-80", "80-100",
          "100-120", "120-140", "140-160", "160-180", "180-200"])
df['absTcorrFisher'] = np.abs(df['TcorrFisher'])

plt.figure(); plt.title('Correlation (z) over distance (traces)')
sns.lineplot(x = 'distance bin', y = 'absTcorrFisher', data = df, hue = 'condition')
plt.ylabel('Fisher-corrected correlation coefficient')
plt.xlabel('Micrometers')





