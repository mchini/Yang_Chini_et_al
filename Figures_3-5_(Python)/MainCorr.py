# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 17:54:53 2019

@author: mchini
"""

#%% load packages and define a few functions

import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import signal
import pickle
from sklearn.metrics.pairwise import euclidean_distances
from scipy.stats import spearmanr
from scipy.stats import zscore
from numba import jit

@jit
def ComputeSpearmanSelfNumba(Matrix):
    for row in np.arange(np.shape(Matrix)[0]):
            temp = Matrix[row, :].argsort()
            Matrix[row, temp] = np.arange(len(temp))
    SpearmanCoeffs = np.corrcoef(Matrix)
    return SpearmanCoeffs

@jit
def ComputeSpearmanPairNumba(Matrix1, Matrix2):
    SpearmanCoeffs = np.zeros((np.shape(Matrix1)[0]))
    for row in np.arange(np.shape(Matrix1)[0]):
            temp = Matrix1[row, :].argsort()
            Matrix1[row, temp] = np.arange(len(temp))
            temp = Matrix2[row, :].argsort()
            Matrix2[row, temp] = np.arange(len(temp))
            SpearmanCoeffs[row] = np.cov(Matrix1[row, :], Matrix2[row, :])[0, 1] \
            / np.sqrt(np.var(Matrix1[row, :]) * np.var(Matrix2[row, :]))
    return SpearmanCoeffs

def ComputeSpearmanSelf(Matrix):
    SpearmanCoeffs = np.zeros((np.shape(Matrix)[0], np.shape(Matrix)[0]))
    for row1 in np.arange(np.shape(Matrix)[0]):
        for row2 in np.arange(row1, np.shape(Matrix)[0]):
            SpearmanCoeffs[row1, row2] = spearmanr(Matrix[row1, ], Matrix[row2])[0]
            SpearmanCoeffs[row2, row1] = SpearmanCoeffs[row1, row2]
    return SpearmanCoeffs
        
def ComputeSpearmanPair(Matrix1, Matrix2):
    SpearmanCoeffs = np.zeros((np.shape(Matrix1)[0]))
    for row in np.arange(np.shape(Matrix1)[0]):
        SpearmanCoeffs[row] = spearmanr(Matrix1[row, ], Matrix2[row])[0]
    return SpearmanCoeffs

#%% load meta data and define a few global variables

path_excel = 'your_folder_here' # path to excel with some meta data
path2save = 'your_folder_here' 
meta_data = pd.read_excel(path_excel)
animals = np.unique(meta_data['Mouse'])

#%% carry out the actual computations

for animal in animals:
    meta_animal = meta_data[meta_data['Mouse'] == animal]
    recordings = meta_animal['Number']
    for idx, recording in enumerate(recordings):
        path_excel_rec = str(meta_data['Folder'][recording] + meta_data['Subfolder'][recording] +
                    str(meta_data['Recording idx'][recording]) + '/suite2p/plane0/')
        stats = np.load(path_excel_rec + '/stat.npy', allow_pickle=True)
        Spikes = np.load(path_excel_rec + '/spks.npy', allow_pickle=True)
        Traces = np.load(path_excel_rec + '/F.npy', allow_pickle=True)
        Npil = np.load(path_excel_rec + '/Fneu.npy', allow_pickle=True)
        Traces = Traces - .7 * Npil
        iscell = np.load(path_excel_rec + '/iscell.npy', allow_pickle=True)
        
        if isinstance(meta_data['Quiet periods'][recording], str):
            good_recording = np.zeros((np.shape(Traces)[1]))
            recording2keep = [int(s) for s in meta_data['Quiet periods'][recording].split(',')]
            begin = recording2keep[0::2]
            ending = recording2keep[1::2]
            for idx_begin in range(int(len(begin))):
                good_recording[begin[idx_begin] : ending[idx_begin]] = 1
        else:
            good_recording = np.ones_like(Spikes[0, :])
        
        good_recording = good_recording > 0
        
        Spikes = Spikes[iscell[:, 0].astype(bool), :]
        Spikes = Spikes[:, good_recording]
        Spikes = zscore(Spikes, axis=1)
        
        Traces = Traces[iscell[:, 0].astype(bool), :]
        Traces = Traces[:, good_recording]
        Traces = zscore(Traces, axis=1)
        
        stats = stats[iscell[:, 0].astype(bool)]
        num_cells = np.shape(Spikes)[0]
        centroid = np.zeros((num_cells, 2))
        CorrSpikes = np.corrcoef(Spikes).flatten()
        CorrTraces = np.corrcoef(Traces).flatten()

        for neuron in np.arange(num_cells):            
            stats_neuron = stats[neuron]
            centroid[neuron, ] = stats_neuron['med']
           
        CorrSpearmanTraces = ComputeSpearmanSelfNumba(Traces)
        CorrSpearmanSpikes = ComputeSpearmanSelfNumba(Spikes)
        CorrNeuropil = ComputeSpearmanPairNumba(Traces, Npil)             
            
        distance_matrix = euclidean_distances(centroid, centroid).flatten()
        
        corr_df = {'condition' : meta_data['Condition'][recording],
                   'CorrSpikes' : CorrSpikes,
                   'CorrTraces' : CorrTraces,
                   'CorrSpikesSpearman' : CorrSpearmanSpikes,
                   'CorrTracessSpearman' : CorrSpearmanTraces,
                   'distance' : distance_matrix}
        
        if not os.path.exists(path2save + str(animal) + '/'):
            os.makedirs(path2save + str(animal) + '/')
        with open(path2save
                  + str(animal) + '/CorrDf' + str(idx), 'wb') as f:
            pickle.dump(corr_df, f)
        print(str(animal) + '-' + str(idx))
        
