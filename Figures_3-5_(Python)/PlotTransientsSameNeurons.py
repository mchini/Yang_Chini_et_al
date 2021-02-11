# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:58:35 2020

@author: mchini
"""

#%% load packages and define some plotting functions

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle


path_excel = 'your_folder_here'
pathNeuronID = 'your_folder_here'
path2load = 'your_folder_here'
path4figures_paired = 'your_folder_here'
path4figures_all = 'your_folder_here'
path4figuresSameCond = 'your_folder_here'
path4figsTransients = 'your_folder_here'
meta_data = pd.read_excel(path_excel)[: 189]
animals = np.unique(meta_data['Mouse'])
sns.set()
sns.set_style("whitegrid")

# plots in figure S7
def plot_kde(data1, data2, xaxis, yaxis, title, prcnt, save, cond1, cond2, path):
    if not prcnt:
        sns.jointplot(data1, data2, kind="kde")
    else:
        min_range = np.percentile(np.concatenate((data1, data2), axis = 0), prcnt[0])
        max_range = np.percentile(np.concatenate((data1, data2), axis = 0), prcnt[1])
        g = sns.jointplot(data1[(data1 > min_range) & (data1 < max_range) &
                                (data2 > min_range) & (data2 < max_range)], 
                          data2[(data1 > min_range) & (data1 < max_range) &
                                (data2 > min_range) & (data2 < max_range)],
                                kind="kde")
    g.set_axis_labels(xaxis, yaxis)
    g.fig.suptitle(title)
    x0, x1 = g.ax_joint.get_xlim()
    y0, y1 = g.ax_joint.get_ylim()
    lims = [max(x0, y0), min(x1, y1)]
    g.ax_joint.plot(lims, lims, ':k')
    if save == 1:
        plt.savefig(path + title + ' ' + cond1 + ' ' + cond2 + '.eps')
        plt.savefig(path + title + ' ' + cond1 + ' ' + cond2 + '.png')
        plt.close()

# compute relative change
def rel_change(data1, data2):
    rel_change = (data1 - data2) / (data1 + data2)
    rel_change = rel_change[~ np.isnan(rel_change)]
    return rel_change

# plot relative change matrix
def plot_rel_change_matrix(df, *args):
    matrix = np.zeros((np.shape(args)[0], np.shape(args)[0]))
    for col_idx1, col1 in enumerate(args):
        for col_idx2, col2 in enumerate(args):
            matrix[col_idx1, col_idx2] = - np.median(rel_change(df[col1], df[col2]))
            matrix[col_idx2, col_idx1] = - matrix[col_idx1, col_idx2]
    fig, ax = plt.subplots()
    plt.imshow(matrix, cmap='RdBu', vmin=-0.4, vmax=0.4)
    plt.colorbar()
    return fig, ax, matrix

# plot several violin plots
def plot_same_cond_across_conds(df, keywords, lims, log, ylabels,
                                titles, xticks, palette, path2save):
    for key_idx, keyword in enumerate(keywords):
        col2plot = [col for col in df.columns if keyword in col] + ['condition']
        df2plot = df[col2plot]
        if lims[key_idx] > 0:
            df2plot = df2plot[df2plot[col2plot[0]] < lims[key_idx]]
        else:
            df2plot = df2plot[df2plot[col2plot[0]] > lims[key_idx]]
        if log[key_idx] > 0:
            df2plot[col2plot[0]] = np.log(df2plot[col2plot[0]])
        plt.figure()
        sns.violinplot('condition', col2plot[0], data=df2plot,
                       cut = 0, palette = palette)
        locs, labels = plt.xticks(); plt.xticks(locs, xticks)
        plt.ylabel(ylabels[key_idx]); plt.title(titles[key_idx])
        plt.savefig(path2save + titles[key_idx] + '.eps')
        plt.savefig(path2save + titles[key_idx] + '.png')
        plt.close()

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
        IDneurons =  np.load(pathNeuronID + str(animal) + '/' + str(idx_rec) + 'bis.npy')
        df1['IDneurons'] = IDneurons.astype(int)
        # this is for same neuron
        if idx_rec == 0:
            df_animal = df1
        else:
            df_animal = [df_animal, df1]
            df_animal = pd.concat(df_animal)
    # find duplicates (neurons that appear twice in the animal)
    duplicates = df_animal.duplicated(subset = ['IDneurons'], keep = False)
    df_dupl = df_animal.loc[duplicates]
    df_dupl['condition'].loc[df_dupl['condition'] == 'quiet'] = 'awa'
    # loop over conditions and if there are duplicates, merge and take median
    for cond_idx, condition in enumerate(np.unique(df_dupl['condition'])):
        condition_df = df_dupl[df_dupl['condition'] == condition]
        condition_df = condition_df.drop(['condition', 'mouse', 'recording',
                                          'decay_no_isol', 'width', 'n_peaks500',
                                          'height500', 'width500'], axis = 1)
        condition_df = condition_df.groupby('IDneurons', as_index = False).median()
        # log-transform height and num_peaks, taking care of avoiding problems with 0s
        condition_df['n_peaks'].loc[condition_df['n_peaks'] == 0] = np.nan
        #condition_df[['n_peaks', 'height']] = np.log10(condition_df[['n_peaks', 'height']])
        condition_df = condition_df.rename(columns = {'decay_isol_10' : 'decay_issol_10'})
        condition_df = condition_df.add_suffix('_' + condition)
        condition_df = condition_df.rename(columns = {'IDneurons' + '_' + condition : 'IDneurons'})
        if cond_idx == 0:
            pairs_df_animal = condition_df        
        else:
            pairs_df_animal = pairs_df_animal.merge(condition_df, on = 'IDneurons', how = 'outer')
    pairs_df_animal['mouse'] = animal
    if idx_animal == 0:
        pairs_df = pairs_df_animal
    else:
        pairs_df = [pairs_df, pairs_df_animal]
        pairs_df = pd.concat(pairs_df)
           

# set quiet condition to awake
df['condition'][df['condition'] == 'quiet'] = 'awa'
# log-transform height and num_peaks, taking care of avoiding problems with 0s
df['n_peaks'][df['n_peaks'] == 0] = np.nan
df[['n_peaks', 'height']] = np.log10(df[['n_peaks', 'height']])

#%% extract "single" same neuron dataframes across conditions

cols_keta = [col for col in pairs_df.columns if 'keta' in col]
cols_fenta = [col for col in pairs_df.columns if 'fenta' in col]
cols_awa = [col for col in pairs_df.columns if 'awa' in col]
cols_iso = [col for col in pairs_df.columns if 'iso' in col]

iso_awa = pairs_df.drop(cols_keta + cols_fenta, axis = 1).dropna()        
iso_fenta = pairs_df.drop(cols_keta + cols_awa, axis = 1).dropna()
iso_keta = pairs_df.drop(cols_awa + cols_fenta, axis = 1).dropna()
fenta_awa = pairs_df.drop(cols_iso + cols_keta, axis = 1).dropna()
keta_awa = pairs_df.drop(cols_iso + cols_fenta, axis = 1).dropna()
keta_fenta = pairs_df.drop(cols_iso + cols_awa, axis = 1).dropna()
all_conds = pairs_df.dropna()

#%% plot the difference matrix

### here height and n_peaks are not logged!!! ###

################## ################## ################## ################## ################## ################## 
     ################## START PLOTTING WITH THE PLOTS THAT WANT DATA IN THE WIDE FORMAT ################## 
################## ################## ################## ################## ################## ################## 

titles_prefix = 'Relative change - '
titles_suffix = ['decay constant', 'number of peaks', 'height']
# set conditions with full name for plotting
conditions = ['awake', 'isoflurane', 'fentanyl', 'ketaxyl']

for index in range(3):
    fig, ax, matrix = plot_rel_change_matrix(pairs_df, cols_awa[index], cols_iso[index],
                                             cols_fenta[index], cols_keta[index])
    ax.set_xticks(np.arange(len(conditions)))
    ax.set_yticks(np.arange(len(conditions)))
    ax.set_xticklabels(conditions)
    ax.set_yticklabels(conditions)
    ax.grid(None)
    title = titles_prefix + titles_suffix[index]
    plt.title(title)
    plt.savefig(path4figures_all + title + '.eps')
    plt.savefig(path4figures_all + title + '.png')
    plt.close()
    
#%% plot transients stuff for same neuron (kde - seaborn)

prcnt = [10, 90]
save = 1
# plot decay of isolated transients
plot_kde(iso_awa['decay_issol_10_awa'], iso_awa['decay_issol_10_iso'],
         'decay awake (ms)', 'decay isoflurane (ms)',
         'Decay constant of isolated transients', prcnt,
         save, 'awake', 'isoflurane', path4figures_paired)

plot_kde(iso_fenta['decay_issol_10_fenta'], iso_fenta['decay_issol_10_iso'],
         'decay fentanyl (ms)', 'decay isoflurane (ms)',
         'Decay constant of isolated transients', prcnt,
         save, 'fentanyl', 'isoflurane', path4figures_paired)

plot_kde(iso_keta['decay_issol_10_keta'], iso_keta['decay_issol_10_iso'],
         'decay ketaxyl (ms)', 'decay isoflurane (ms)',
         'Decay constant of isolated transients', prcnt,
         save, 'ketaxyl', 'isoflurane', path4figures_paired)

plot_kde(fenta_awa['decay_issol_10_awa'], fenta_awa['decay_issol_10_fenta'],
         'decay awake (ms)', 'decay fentanyl (ms)',
         'Decay constant of isolated transients', prcnt,
         save, 'awake', 'fentanyl', path4figures_paired)

plot_kde(keta_awa['decay_issol_10_awa'], keta_awa['decay_issol_10_keta'],
         'decay awake (ms)', 'decay ketaxyl (ms)',
         'Decay constant of isolated transients', prcnt,
         save, 'awake', 'ketaxyl', path4figures_paired)

plot_kde(keta_fenta['decay_issol_10_keta'], keta_fenta['decay_issol_10_fenta'],
         'decay ketaxyl (ms)', 'decay fentanyl (ms)',
         'Decay constant of isolated transients', prcnt,
         save, 'ketaxyl', 'fentanyl', path4figures_paired)

# plot height of transients
plot_kde(iso_awa['height_awa'], iso_awa['height_iso'],
         'log of height awake (A.U.)', 'log of height isoflurane (A.U.)',
         'Log of transients height', prcnt,
         save, 'awake', 'isoflurane', path4figures_paired)

plot_kde(iso_fenta['height_fenta'], iso_fenta['height_iso'],
         'log of height fentanyl (A.U.)', 'log of height isoflurane (A.U.)',
         'Log of transients height', prcnt,
         save, 'fentanyl', 'isoflurane', path4figures_paired)

plot_kde(iso_keta['height_keta'], iso_keta['height_iso'],
         'log of height ketaxyl (A.U.)', 'log of height isoflurane (A.U.)',
         'Log of transients height', prcnt,
         save, 'ketaxyl', 'isoflurane', path4figures_paired)

plot_kde(fenta_awa['height_awa'], fenta_awa['height_fenta'],
         'log of height awake (A.U.)', 'log of height fentanyl (A.U.)',
         'Log of transients height', prcnt,
         save, 'awake', 'fentanyl', path4figures_paired)

plot_kde(keta_awa['height_awa'], keta_awa['height_keta'],
         'log of height awake (A.U.)', 'log of height ketaxyl (A.U.)',
         'Log of transients height', prcnt,
         save, 'awake', 'ketaxyl', path4figures_paired)

plot_kde(keta_fenta['height_keta'], keta_fenta['height_fenta'],
         'log of height ketaxyl (A.U.)', 'log of height fentanyl (A.U.)',
         'Log of transients height', prcnt,
         save, 'ketaxyl', 'fentanyl', path4figures_paired)

# plot number of transients
plot_kde(iso_awa['n_peaks_awa'], iso_awa['n_peaks_iso'],
         'log of n_peaks awake (n° peaks / minute)',
         'log of n_peaks isoflurane (n° peaks / minute)',
         'Log of transients n_peaks', prcnt,
         save, 'awake', 'isoflurane', path4figures_paired)

plot_kde(iso_fenta['n_peaks_fenta'], iso_fenta['n_peaks_iso'],
         'log of n_peaks fentanyl (n° peaks / minute)',
         'log of n_peaks isoflurane (n° peaks / minute)',
         'Log of transients n_peaks', prcnt,
         save, 'fentanyl', 'isoflurane', path4figures_paired)

plot_kde(iso_keta['n_peaks_keta'], iso_keta['n_peaks_iso'],
         'log of n_peaks ketaxyl (n° peaks / minute)',
         'log of n_peaks isoflurane (n° peaks / minute)',
         'Log of transients n_peaks', prcnt,
         save, 'ketaxyl', 'isoflurane', path4figures_paired)

plot_kde(fenta_awa['n_peaks_awa'], fenta_awa['n_peaks_fenta'],
         'log of n_peaks awake (n° peaks / minute)',
         'log of n_peaks fentanyl (n° peaks / minute)',
         'Log of transients n_peaks', prcnt,
         save, 'awake', 'fentanyl', path4figures_paired)

plot_kde(keta_awa['n_peaks_awa'], keta_awa['n_peaks_keta'],
         'log of n_peaks awake (n° peaks / minute)',
         'log of n_peaks ketaxyl (n° peaks / minute)',
         'Log of transients n_peaks', prcnt,
         save, 'awake', 'ketaxyl', path4figures_paired)

plot_kde(keta_fenta['n_peaks_keta'], keta_fenta['n_peaks_fenta'],
         'log of n_peaks ketaxyl (n° peaks / minute)',
         'log of n_peaks fentanyl (n° peaks / minute)',
         'Log of transients n_peaks', prcnt,
         save, 'ketaxyl', 'fentanyl', path4figures_paired)


#%% prepare paired datasets

# convert them from wide to long
iso_awa['id'] = np.arange(0, np.shape(iso_awa)[0])
iso_awa = pd.wide_to_long(iso_awa, stubnames = ['decay_issol_10', 'n_peaks', 'height'],
                        i = 'id', j = 'condition', sep = '_', suffix = '\D+')
iso_awa.index = iso_awa.index.droplevel('id')
iso_awa = iso_awa.rename(columns = {'decay_issol_10' : 'decay'})

iso_fenta['id'] = np.arange(0, np.shape(iso_fenta)[0])
iso_fenta = pd.wide_to_long(iso_fenta, stubnames = ['decay_issol_10', 'n_peaks', 'height'],
                        i = 'id', j = 'condition', sep = '_', suffix = '\D+')
iso_fenta.index = iso_fenta.index.droplevel('id')
iso_fenta = iso_fenta.rename(columns = {'decay_issol_10' : 'decay'})

iso_keta['id'] = np.arange(0, np.shape(iso_keta)[0])
iso_keta = pd.wide_to_long(iso_keta, stubnames = ['decay_issol_10', 'n_peaks', 'height'],
                        i = 'id', j = 'condition', sep = '_', suffix = '\D+')
iso_keta.index = iso_keta.index.droplevel('id')
iso_keta = iso_keta.rename(columns = {'decay_issol_10' : 'decay'})

fenta_awa['id'] = np.arange(0, np.shape(fenta_awa)[0])
fenta_awa = pd.wide_to_long(fenta_awa, stubnames = ['decay_issol_10', 'n_peaks', 'height'],
                        i = 'id', j = 'condition', sep = '_', suffix = '\D+')
fenta_awa.index = fenta_awa.index.droplevel('id')
fenta_awa = fenta_awa.rename(columns = {'decay_issol_10' : 'decay'})

keta_awa['id'] = np.arange(0, np.shape(keta_awa)[0])
keta_awa = pd.wide_to_long(keta_awa, stubnames = ['decay_issol_10', 'n_peaks', 'height'],
                        i = 'id', j = 'condition', sep = '_', suffix = '\D+')
keta_awa.index = keta_awa.index.droplevel('id')
keta_awa = keta_awa.rename(columns = {'decay_issol_10' : 'decay'})

keta_fenta['id'] = np.arange(0, np.shape(keta_fenta)[0])
keta_fenta = pd.wide_to_long(keta_fenta, stubnames = ['decay_issol_10', 'n_peaks', 'height'],
                        i = 'id', j = 'condition', sep = '_', suffix = '\D+')
keta_fenta.index = keta_fenta.index.droplevel('id')
keta_fenta = keta_fenta.rename(columns = {'decay_issol_10' : 'decay'})

all_conds['id'] = np.arange(0, np.shape(all_conds)[0])
all_conds = pd.wide_to_long(all_conds, stubnames = ['decay_issol_10', 'n_peaks', 'height'],
                        i = 'id', j = 'condition', sep = '_', suffix = '\D+')
all_conds.index = all_conds.index.droplevel('id')
all_conds = all_conds.rename(columns = {'decay_issol_10' : 'decay'})

# log height and n_peaks
iso_awa['n_peaks'][iso_awa['n_peaks'] == 0] = np.nan
iso_awa[['n_peaks', 'height']] = np.log10(iso_awa[['n_peaks', 'height']])
iso_fenta['n_peaks'][iso_fenta['n_peaks'] == 0] = np.nan
iso_fenta[['n_peaks', 'height']] = np.log10(iso_fenta[['n_peaks', 'height']])
iso_keta['n_peaks'][iso_keta['n_peaks'] == 0] = np.nan
iso_keta[['n_peaks', 'height']] = np.log10(iso_keta[['n_peaks', 'height']])
fenta_awa['n_peaks'][fenta_awa['n_peaks'] == 0] = np.nan
fenta_awa[['n_peaks', 'height']] = np.log10(fenta_awa[['n_peaks', 'height']])
keta_awa['n_peaks'][keta_awa['n_peaks'] == 0] = np.nan
keta_awa[['n_peaks', 'height']] = np.log10(keta_awa[['n_peaks', 'height']])
keta_fenta['n_peaks'][keta_fenta['n_peaks'] == 0] = np.nan
keta_fenta[['n_peaks', 'height']] = np.log10(keta_fenta[['n_peaks', 'height']])
all_conds['n_peaks'][all_conds['n_peaks'] == 0] = np.nan
all_conds[['n_peaks', 'height']] = np.log10(all_conds[['n_peaks', 'height']])


#%% plot transients stuff

# extract palette and set color according to what you prefer
current_palette = np.array(sns.color_palette())
colors_violin = sns.color_palette(current_palette[np.r_[0, 2, 1, 3], :])
conditions = np.unique(df['condition'])
conditions = conditions[np.r_[0, 2, 1, 3]]

plt.figure(); sns.violinplot(x = 'condition', y = 'decay_isol_10',
          data = df.loc[df['decay_isol_10'] < 1800], cut = 0,
          order = conditions, palette = colors_violin)
plt.ylim([0, 1000])
title = 'Calcium transient decay constant'
plt.ylabel('Time (ms)'); plt.title(title)
plt.savefig(path4figsTransients + title + '.eps')
plt.savefig(path4figsTransients + title + '.png')
plt.close()

# log transform number of calcium events
plt.figure(); sns.violinplot(x = 'condition', y = 'n_peaks', cut = 0, 
          order = conditions, data = df[np.isfinite(df['n_peaks'])],
          palette = colors_violin)
title = 'N° of calcium peaks'
plt.ylabel('Log of calcium peaks / minute'); plt.title(title)
plt.savefig(path4figsTransients + title + '.eps')
plt.savefig(path4figsTransients + title + '.png')
plt.close()

# log transform height of calcium events
plt.figure(); sns.violinplot(x = 'condition', y = 'height', cut = 0, 
          order = conditions, data = df[np.isfinite(df['height'])],
          palette = colors_violin)
title = 'Height of transients'
plt.ylabel('Log of arbitrary units'); plt.title(title)
plt.ylim([2.3, 3.2])
plt.savefig(path4figsTransients + title + '.eps')
plt.savefig(path4figsTransients + title + '.png')
plt.close()
  
#%% plot all the summary plots that are done for the entire dataframe but now only for "all_conds" dataframe

# extract palette and set color according to what you prefer
current_palette = np.array(sns.color_palette())
colors_violin = sns.color_palette(current_palette[np.r_[0, 2, 1, 3], :])

######## violin plots ########

plt.figure(); sns.violinplot(cut = 0, y = 'n_peaks', x = 'condition',
          data = all_conds.reset_index(), order = conditions,
          palette = colors_violin)
plt.ylabel('Log of calcium peaks / minute'); plt.title('N° of calcium peaks')
locs, labels = plt.xticks(); plt.xticks(locs, ('Awa', 'Iso', 'MMF', 'Ketaxyl'))
plt.savefig(path4figures_all + 'num of peaks - all conditions.eps')
plt.savefig(path4figures_all + 'num of peaks - all conditions.png')
plt.close()

plt.figure(); sns.violinplot(cut = 0, y = 'height', x = 'condition',
          data = all_conds.loc[all_conds['height'] < 1400].reset_index(),
          order = conditions, palette = colors_violin)
plt.ylabel('Log of arbitrary units'); plt.title('Height transients')
plt.ylim([2.3, 3.4])
locs, labels = plt.xticks(); plt.xticks(locs, ('Awa', 'Iso', 'MMF', 'Ketaxyl'))
plt.savefig(path4figures_all + 'height of peaks - all conditions.eps')
plt.savefig(path4figures_all + 'height of peaks - all conditions.png')
plt.close()

# calcium transients decay constant
plt.figure(); sns.violinplot(y = 'decay', x = 'condition',
          data = all_conds.reset_index(), order = conditions,
          cut = 0, palette = colors_violin)
plt.ylabel('Time (ms)'); plt.title('Calcium transient decay constant')
plt.ylim([0, 1200])
locs, labels = plt.xticks(); plt.xticks(locs, ('Awa', 'Iso', 'MMF', 'Ketaxyl'))
plt.savefig(path4figures_all + 'decay of peaks - all conditions.eps')
plt.savefig(path4figures_all + 'decay of peaks - all conditions.png')
plt.close()