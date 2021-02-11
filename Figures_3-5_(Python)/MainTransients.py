# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:38:31 2020

@author: mchini
"""

#%% load packages

import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
import pickle
from sys import path
path.append('your_path2_OASIS_toolbox')
from oasis.functions import deconvolve

path_excel = 'your_path_here'
path2save = 'your_path_here'
meta_data = pd.read_excel(path_excel)
animals = np.unique(meta_data['Mouse'])
repeat_calc = 0
sns.set()
sns.set_style("whitegrid")
fs = 30 # sampling rate of the recording

#%% create dataframe with xcorr values and unique pair ID

for animal in animals:
    meta_animal = meta_data[meta_data['Mouse'] == animal]
    recordings = meta_animal['Number']
    for idx, recording in enumerate(recordings):
        path_excel_rec = str(meta_data['Folder'][recording] + meta_data['Subfolder'][recording] +
                    str(meta_data['Recording idx'][recording]) + '/suite2p/plane0/')
        stats = np.load(path_excel_rec + '/stat.npy', allow_pickle=True)
        Traces = np.load(path_excel_rec + '/F.npy', allow_pickle=True)
        Npil = np.load(path_excel_rec + '/Fneu.npy', allow_pickle=True)
        Traces = Traces - .7 * Npil
        iscell = np.load(path_excel_rec + '/iscell.npy', allow_pickle=True)
        
        if isinstance(meta_data['Quiet periods'][recording], str):
            good_recording = np.empty((9000))
            recording2keep = [int(s) for s in meta_data['Quiet periods'][recording].split(',')]
            begin = recording2keep[0::2]
            ending = recording2keep[1::2]
            for idx_begin in range(int(len(begin))):
                good_recording[begin[idx_begin] : ending[idx_begin]] = 1
        else:
            good_recording = np.ones_like(Traces[0, :])
        
        Traces = Traces[iscell[:, 0].astype(bool), :]
        Traces = Traces - np.tile(np.expand_dims(np.median(Traces, axis=1), axis=1),
                                  (1, np.shape(Traces)[1]))
        
        num_cells = np.shape(Traces)[0]
        decay_isol = np.zeros((num_cells))
        n_peaks = np.zeros((num_cells))
        height = np.zeros((num_cells))
        trace_len = np.shape(Traces)[1] / (fs * 60) # in minutes       
        
        for neuron in np.arange(num_cells):
            
            if np.all(np.isnan(Traces[neuron])):
                decay_isol[neuron] = np.nan
            else:
                _, _, _, decay_neuron_isolated10, _ = deconvolve(np.double(Traces[neuron, ]),
                                                                 penalty = 0, optimize_g = 10)
                decay_isol[neuron] = - 1 / (fs * np.log(decay_neuron_isolated10))
                
            _, peaks = signal.find_peaks(Traces[neuron, ], height = 200,
                                         distance = 10, prominence = 200,
                                         width = (None, None),
                                         rel_height = 0.9)
            n_peaks[neuron] = len(peaks['peak_heights']) / trace_len
            
            if n_peaks[neuron] > 0:
                height[neuron, ] = np.median(peaks['peak_heights'])
            else:
                height[neuron, ] = np.nan

        transients_df = {'condition' : meta_data['Condition'][recording],
                        'decay_isol_10' : decay_isol,
                        'n_peaks' : n_peaks,
                        'height' : height,
                        }
        
        if not os.path.exists(path2save + str(animal) + '/'):
            os.makedirs(path2save + str(animal) + '/')
        with open(path2save + str(animal) + '/transients_df' + str(idx), 'wb') as f:
            pickle.dump(transients_df, f)
        print(str(animal) + '-' + str(idx))
        
