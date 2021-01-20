#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 18:01:07 2021

@author: aformozov
"""

import numpy as np

import numpy.ma as ma

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import os


import sys

if sys.platform=='linux':
    data_path = "/media/andrey/My Passport/_routine-analysis-pipeline_concatenate/"
else:
    data_path = "Q:/Andrey/DataAnalysis/_routine-analysis-pipeline_concatenate/"
    
########## STYLE ###########

plt.rcParams['figure.figsize'] = [12, 8]
sns.set_context("paper", rc={"font.size":12,"axes.titlesize":12,"axes.labelsize":15})  
sns.set({"xtick.major.size": 12, "ytick.major.size": 12, 'xtick.labelsize': 'large', 'ytick.labelsize': 'large',
 'legend.fontsize': 'large'},    style="ticks")


df_estimators_old_mmf = pd.read_pickle(data_path + "df_estimators_TS_C_MMF_F0_M0_n_peaks_baseline.pkl")
df_estimators_new_mmf = pd.read_pickle(data_path + "df_estimators_TS_C_MMF_8235_8237_8238_n_peaks_baseline.pkl")
df_estimators_mmf = pd.concat([df_estimators_new_mmf,df_estimators_old_mmf])

df_estimators_new_iso = pd.read_pickle(data_path + "df_estimators_TS_C_ISO_8235_8237_8238_n_peaks_baseline.pkl")
df_estimators_old_iso = pd.read_pickle(data_path + "df_estimators_TS_C_ISO_F0_M0_n_peaks_baseline.pkl")
df_estimators_iso = pd.concat([df_estimators_new_iso,df_estimators_old_iso])

df_estimators_new_keta = pd.read_pickle(data_path + "df_estimators_TS_C_KETA_8235_8237_8238_n_peaks_baseline.pkl")
df_estimators_old_keta = pd.read_pickle(data_path + "df_estimators_TS_C_KETA_F0_M3_n_peaks_baseline.pkl")
df_estimators_keta = pd.concat([df_estimators_new_keta,df_estimators_old_keta])


######## TODO CHECK TIMING ########

condition_to_time_mmf = {"awake":0,"mmf":30,"post30":60,"post60":90,"post120":150,"post180":210,"post240":270,"post300":330,"post360":390}
condition_to_time_keta = {"awake":0,"keta":30,"keta30":60,"keta60":90,"keta120":150,"keta180":210,"keta240":270,"keta300":330,"keta360":390}
condition_to_time_iso = {"awake":0,"iso":30,"rec30":60,"rec60":90,"rec120":150,"rec180":210,"rec240":270,"rec300":330,"rec360":390}


df_estimators_mmf['time'] = [condition_to_time_mmf[x] for x in df_estimators_mmf['condition']]
df_estimators_keta['time'] = [condition_to_time_keta[x] for x in df_estimators_keta['condition']]
df_estimators_iso['time'] = [condition_to_time_iso[x] for x in df_estimators_iso['condition']]


color_mmf = (245/255,143/255,32/255)
color_keta = (181./255,34./255,48./255)
color_iso = (143./255,39./255,143./255)


################################################################################################################
################################################################################################################

pd.set_option('display.max_rows', None)

### CALCULATE MODULATION INDEX N_PEAKS


df_estimators_mmf['n_peaks_mod'] = df_estimators_mmf['n_peaks']

for animal in ['TS_C_MMF_8235', 'TS_C_MMF_8237', 'TS_C_MMF_8238', 'TS_C_MMF_F0','TS_C_MMF_M0']: 
    
    sel = (df_estimators_mmf.n_peaks > 0.3)&(df_estimators_mmf.decay_isol > 1/30.)&(df_estimators_mmf.decay_isol < 10)    
    sel_animal = (df_estimators_mmf.animal == animal)
    mmf_awake_median = df_estimators_mmf[sel&sel_animal&(df_estimators_mmf.condition == 'awake')]['n_peaks_mod'].median()
    yyy = df_estimators_mmf.loc[sel&sel_animal, 'n_peaks_mod' ]
    yyy =  (yyy - mmf_awake_median)/(yyy + mmf_awake_median)
    df_estimators_mmf.loc[sel&sel_animal, 'n_peaks_mod']  = yyy
    df_estimators_mmf.loc[(~sel)&sel_animal, 'n_peaks_mod']  = np.nan
    
    
print( df_estimators_mmf.loc[sel&sel_animal]['n_peaks_mod'].median() )

df_estimators_iso['n_peaks_mod'] = df_estimators_iso['n_peaks']

for animal in ['TS_C_ISO_8235', 'TS_C_ISO_8237', 'TS_C_ISO_8238', 'TS_C_ISO_M0','TS_C_ISO_F0']: 

    sel = (df_estimators_iso.n_peaks > 0.3)&(df_estimators_iso.decay_isol > 1/30.)&(df_estimators_iso.decay_isol < 10) 
    sel_animal = (df_estimators_iso.animal == animal)
    iso_awake_median = df_estimators_iso[sel&sel_animal&(df_estimators_iso.condition == 'awake')]['n_peaks_mod'].median()
    yyy = df_estimators_iso.loc[sel&sel_animal, 'n_peaks_mod' ]
    yyy =  (yyy - iso_awake_median)/(yyy + iso_awake_median)
    df_estimators_iso.loc[sel&sel_animal, 'n_peaks_mod']  = yyy
    df_estimators_iso.loc[(~sel)&sel_animal, 'n_peaks_mod']  = np.nan


df_estimators_keta['n_peaks_mod'] = df_estimators_keta['n_peaks']

for animal in ['TS_C_KETA_8235', 'TS_C_KETA_8237', 'TS_C_KETA_8238', 'TS_C_KETA_F0','TS_C_KETA_M3']: 

    sel = (df_estimators_keta.n_peaks > 0.3)&(df_estimators_keta.decay_isol > 1/30.)&(df_estimators_keta.decay_isol < 10)
    sel_animal = (df_estimators_keta.animal == animal)
    keta_awake_median = df_estimators_keta[sel&sel_animal&(df_estimators_keta.condition == 'awake')]['n_peaks_mod'].median()
    yyy = df_estimators_keta.loc[sel&sel_animal, 'n_peaks_mod' ]
    yyy =  (yyy - keta_awake_median)/(yyy + keta_awake_median)
    df_estimators_keta.loc[sel&sel_animal, 'n_peaks_mod']  = yyy
    df_estimators_keta.loc[(~sel)&sel_animal, 'n_peaks_mod']  = np.nan


################################################################################################################

### CALCULATE MODULATION INDEX AMPLITUDE

df_estimators_mmf['height.median_mod'] = df_estimators_mmf['height.median']

for animal in ['TS_C_MMF_8235', 'TS_C_MMF_8237', 'TS_C_MMF_8238', 'TS_C_MMF_F0','TS_C_MMF_M0']: 
      
    sel = (df_estimators_mmf.n_peaks > 0.3)&(df_estimators_mmf.decay_isol > 1/30.)&(df_estimators_mmf.decay_isol < 10)  
    sel_animal = (df_estimators_mmf.animal == animal)
    mmf_awake_median = df_estimators_mmf[sel&sel_animal&(df_estimators_mmf.condition == 'awake')]['height.median_mod'].median()
    yyy = df_estimators_mmf.loc[sel&sel_animal, 'height.median_mod' ]
    yyy =  (yyy - mmf_awake_median)/(yyy + mmf_awake_median)
    df_estimators_mmf.loc[sel&sel_animal, 'height.median_mod']  = yyy
    df_estimators_mmf.loc[(~sel)&sel_animal, 'height.median_mod']  = np.nan
    
print( df_estimators_mmf.loc[sel&sel_animal]['n_peaks_mod'].median() )


df_estimators_iso['height.median_mod'] = df_estimators_iso['height.median']

for animal in ['TS_C_ISO_8235', 'TS_C_ISO_8237', 'TS_C_ISO_8238', 'TS_C_ISO_M0','TS_C_ISO_F0']: 

    sel = (df_estimators_iso.n_peaks > 0.3)&(df_estimators_iso.decay_isol > 1/30.)&(df_estimators_iso.decay_isol < 10)   
    sel_animal = (df_estimators_iso.animal == animal)
    iso_awake_median = df_estimators_iso[sel&sel_animal&(df_estimators_iso.condition == 'awake')]['height.median_mod'].median()
    yyy = df_estimators_iso.loc[sel&sel_animal, 'height.median_mod' ]
    yyy =  (yyy - iso_awake_median)/(yyy + iso_awake_median)
    df_estimators_iso.loc[sel&sel_animal, 'height.median_mod']  = yyy
    df_estimators_iso.loc[(~sel)&sel_animal, 'height.median_mod']  = np.nan

df_estimators_keta['height.median_mod'] = df_estimators_keta['height.median']

for animal in ['TS_C_KETA_8235', 'TS_C_KETA_8237', 'TS_C_KETA_8238', 'TS_C_KETA_F0','TS_C_KETA_M3']: 
      
    sel = (df_estimators_keta.n_peaks > 0.3)&(df_estimators_keta.decay_isol > 1/30.)&(df_estimators_keta.decay_isol < 10)
    sel_animal = (df_estimators_keta.animal == animal)
    keta_awake_median = df_estimators_keta[sel&sel_animal&(df_estimators_keta.condition == 'awake')]['height.median_mod'].median()
    yyy = df_estimators_keta.loc[sel&sel_animal, 'height.median_mod' ]
    yyy =  (yyy - keta_awake_median)/(yyy + keta_awake_median)
    df_estimators_keta.loc[sel&sel_animal, 'height.median_mod']  = yyy
    df_estimators_keta.loc[(~sel)&sel_animal, 'height.median_mod']  = np.nan

################################################################################################################

### CALCULATE MODULATION INDEX DECAY CONSTANT

df_estimators_mmf['decay_isol_mod'] = df_estimators_mmf['decay_isol']

for animal in ['TS_C_MMF_8235', 'TS_C_MMF_8237', 'TS_C_MMF_8238', 'TS_C_MMF_F0','TS_C_MMF_M0']: 
    
    sel = (df_estimators_mmf.n_peaks > 0.3)&(df_estimators_mmf.decay_isol > 1/30.)&(df_estimators_mmf.decay_isol < 10)
    sel_animal = (df_estimators_mmf.animal == animal)
    mmf_awake_median = df_estimators_mmf[sel&sel_animal&(df_estimators_mmf.condition == 'awake')]['decay_isol_mod'].median()
    yyy = df_estimators_mmf.loc[sel&sel_animal, 'decay_isol_mod' ]
    yyy =  (yyy - mmf_awake_median)/(yyy + mmf_awake_median)
    df_estimators_mmf.loc[sel&sel_animal, 'decay_isol_mod']  = yyy
    df_estimators_mmf.loc[(~sel)&sel_animal, 'decay_isol_mod']  = np.nan

print( df_estimators_mmf.loc[sel&sel_animal]['n_peaks_mod'].median() )


df_estimators_iso['decay_isol_mod'] = df_estimators_iso['decay_isol']

for animal in ['TS_C_ISO_8235', 'TS_C_ISO_8237', 'TS_C_ISO_8238', 'TS_C_ISO_M0','TS_C_ISO_F0']: 

    sel = (df_estimators_iso.n_peaks > 0.3)&(df_estimators_iso.decay_isol > 1/30.)&(df_estimators_iso.decay_isol < 10)
    sel_animal = (df_estimators_iso.animal == animal)
    iso_awake_median = df_estimators_iso[sel&sel_animal&(df_estimators_iso.condition == 'awake')]['decay_isol_mod'].median()
    yyy = df_estimators_iso.loc[sel&sel_animal, 'decay_isol_mod' ]
    yyy =  (yyy - iso_awake_median)/(yyy + iso_awake_median)
    df_estimators_iso.loc[sel&sel_animal, 'decay_isol_mod']  = yyy
    df_estimators_iso.loc[(~sel)&sel_animal, 'decay_isol_mod']  = np.nan

df_estimators_keta['decay_isol_mod'] = df_estimators_keta['decay_isol']

for animal in ['TS_C_KETA_8235', 'TS_C_KETA_8237', 'TS_C_KETA_8238', 'TS_C_KETA_F0','TS_C_KETA_M3']: 

    sel = (df_estimators_keta.n_peaks > 0.3)&(df_estimators_keta.decay_isol > 1/30.)&(df_estimators_keta.decay_isol < 10)
    sel_animal = (df_estimators_keta.animal == animal)
    keta_awake_median = df_estimators_keta[sel&sel_animal&(df_estimators_keta.condition == 'awake')]['decay_isol_mod'].median()
    yyy = df_estimators_keta.loc[sel&sel_animal, 'decay_isol_mod' ]
    yyy =  (yyy - keta_awake_median)/(yyy + keta_awake_median)
    df_estimators_keta.loc[sel&sel_animal, 'decay_isol_mod']  = yyy
    df_estimators_keta.loc[(~sel)&sel_animal, 'decay_isol_mod']  = np.nan

###################################################################################################################################################
################################# NEURON IDs for Stats ############################################################################################
###################################################################################################################################################

n_iso_8235 = max(df_estimators_iso[ df_estimators_iso.animal == 'TS_C_ISO_8235']['neuronID']) + 1 
n_iso_8237 = max(df_estimators_iso[ df_estimators_iso.animal == 'TS_C_ISO_8237']['neuronID']) + 1
n_iso_8238 = max(df_estimators_iso[ df_estimators_iso.animal == 'TS_C_ISO_8238']['neuronID']) + 1
n_iso_F0 = max(df_estimators_iso[ df_estimators_iso.animal == 'TS_C_ISO_F0']['neuronID']) + 1

df_estimators_iso.loc[df_estimators_iso.animal == 'TS_C_ISO_8235','neuronID'] = df_estimators_iso.loc[df_estimators_iso.animal == 'TS_C_ISO_8235','neuronID']
df_estimators_iso.loc[df_estimators_iso.animal == 'TS_C_ISO_8237','neuronID'] = n_iso_8235  + df_estimators_iso.loc[df_estimators_iso.animal == 'TS_C_ISO_8237','neuronID']
df_estimators_iso.loc[df_estimators_iso.animal == 'TS_C_ISO_8238','neuronID'] = n_iso_8235  + n_iso_8237 + df_estimators_iso.loc[df_estimators_iso.animal == 'TS_C_ISO_8238','neuronID']
df_estimators_iso.loc[df_estimators_iso.animal == 'TS_C_ISO_F0','neuronID'] =  n_iso_8235 + n_iso_8237 + n_iso_8238 + df_estimators_iso.loc[df_estimators_iso.animal == 'TS_C_ISO_F0','neuronID']
df_estimators_iso.loc[df_estimators_iso.animal == 'TS_C_ISO_M0','neuronID'] =   n_iso_8235 + n_iso_8237 + n_iso_8238 + n_iso_F0 + df_estimators_iso.loc[df_estimators_iso.animal == 'TS_C_ISO_M0','neuronID']

##################################################################################################################################################

n_keta_8235 = max(df_estimators_keta[ df_estimators_keta.animal == 'TS_C_KETA_8235']['neuronID']) + 1 
n_keta_8237 = max(df_estimators_keta[ df_estimators_keta.animal == 'TS_C_KETA_8237']['neuronID']) + 1
n_keta_8238 = max(df_estimators_keta[ df_estimators_keta.animal == 'TS_C_KETA_8238']['neuronID']) + 1
n_keta_F0 = max(df_estimators_keta[ df_estimators_keta.animal == 'TS_C_KETA_F0']['neuronID']) + 1

df_estimators_keta.loc[df_estimators_keta.animal == 'TS_C_KETA_8235','neuronID'] = df_estimators_keta.loc[df_estimators_keta.animal == 'TS_C_KETA_8235','neuronID']
df_estimators_keta.loc[df_estimators_keta.animal == 'TS_C_KETA_8237','neuronID'] = n_keta_8235 + df_estimators_keta.loc[df_estimators_keta.animal == 'TS_C_KETA_8237','neuronID']
df_estimators_keta.loc[df_estimators_keta.animal == 'TS_C_KETA_8238','neuronID'] = n_keta_8235 + n_keta_8237 + df_estimators_keta.loc[df_estimators_keta.animal == 'TS_C_KETA_8238','neuronID']
df_estimators_keta.loc[df_estimators_keta.animal == 'TS_C_KETA_F0','neuronID'] =   n_keta_8235 + n_keta_8237 + n_keta_8238 + df_estimators_keta.loc[df_estimators_keta.animal == 'TS_C_KETA_F0','neuronID']
df_estimators_keta.loc[df_estimators_keta.animal == 'TS_C_KETA_M3','neuronID'] =   n_keta_8235 + n_keta_8237 + n_keta_8238 + n_keta_F0 + df_estimators_keta.loc[df_estimators_keta.animal == 'TS_C_KETA_M3','neuronID']

#################################################################################################################################################

n_mmf_8235 = max(df_estimators_mmf[ df_estimators_mmf.animal == 'TS_C_MMF_8235']['neuronID']) + 1 
n_mmf_8237 = max(df_estimators_mmf[ df_estimators_mmf.animal == 'TS_C_MMF_8237']['neuronID']) + 1
n_mmf_8238 = max(df_estimators_mmf[ df_estimators_mmf.animal == 'TS_C_MMF_8238']['neuronID']) + 1
n_mmf_F0 = max(df_estimators_mmf[ df_estimators_mmf.animal == 'TS_C_MMF_F0']['neuronID']) + 1                                                                       
                                                                               
                                                                              
df_estimators_mmf.loc[df_estimators_mmf.animal == 'TS_C_MMF_8235','neuronID'] = df_estimators_mmf.loc[df_estimators_mmf.animal == 'TS_C_MMF_8235','neuronID']
df_estimators_mmf.loc[df_estimators_mmf.animal == 'TS_C_MMF_8237','neuronID'] = n_mmf_8235 + df_estimators_mmf.loc[df_estimators_mmf.animal == 'TS_C_MMF_8237','neuronID']
df_estimators_mmf.loc[df_estimators_mmf.animal == 'TS_C_MMF_8238','neuronID'] = n_mmf_8235 + n_mmf_8237 + df_estimators_mmf.loc[df_estimators_mmf.animal == 'TS_C_MMF_8238','neuronID']
df_estimators_mmf.loc[df_estimators_mmf.animal == 'TS_C_MMF_F0','neuronID'] =   n_mmf_8235 + n_mmf_8237 + n_mmf_8238 + df_estimators_mmf.loc[df_estimators_mmf.animal == 'TS_C_MMF_F0','neuronID']
df_estimators_mmf.loc[df_estimators_mmf.animal == 'TS_C_MMF_M3','neuronID'] =   n_mmf_8235 + n_mmf_8237 + n_mmf_8238 + n_mmf_F0 + df_estimators_mmf.loc[df_estimators_mmf.animal == 'TS_C_MMF_M3','neuronID']


###################################################################### PLOT ##############################################################################
##########################################################################################################################################################

sns.lineplot(x='time', y='n_peaks_mod', markers='o',lw=3,estimator=np.median,ci=95, color=color_keta,data=df_estimators_keta.dropna(),label='Keta/Xyl')
sns.lineplot(x='time', y='n_peaks_mod', markers='o',lw=3,estimator=np.median,ci=95, color=color_mmf,data=df_estimators_mmf.dropna(),label='MMF')
sns.lineplot(x='time', y='n_peaks_mod', markers='o',lw=3,estimator=np.median,ci=95, color=color_iso,data=df_estimators_iso.dropna(),label='Iso')

plt.legend(fontsize=15)
plt.ylabel("modulation index (number of transients)",fontsize=15)
#plt.title("modulation index (transients)",fontsize=22)
plt.xlabel("Time, [min]", fontsize=15)
plt.xlim([0.0, 360.0])


plt.savefig("./n_peaks_mod.svg", format = 'svg', dpi=300)
#plt.savefig('height.median_norm.eps', format='eps')
plt.savefig('./n_peaks_mod.png', format='png')
    
### AMPLITUDE

sns.lineplot(x='time', y='height.median_mod', lw=3, estimator=np.median, ci=95, color=color_mmf, data=df_estimators_mmf.dropna(), label='MMF')
sns.lineplot(x='time', y='height.median_mod', markers='o',lw=3,estimator=np.median,ci=95, color=color_keta,data=df_estimators_keta.dropna(),label='Keta/Xyl')
sns.lineplot(x='time', y='height.median_mod', markers='o',lw=3,estimator=np.median,ci=95, color=color_iso,data=df_estimators_iso.dropna(),label='Iso')

plt.legend(fontsize=15)
#plt.title("modulation index (transients)",fontsize=22)
plt.ylabel("modulation index (amplitude)",fontsize=15)
plt.xlabel("Time, [min]", fontsize=15)
#plt.ylim([0.5,1.5])
plt.xlim([0,360])

plt.savefig("./height.median_mod.svg", format = 'svg', dpi=300)
#plt.savefig('height.median_norm.eps', format='eps')
plt.savefig('./height.median_mod.png', format='png')


#(df_estimators_keta.n_peaks > 0.3)&(df_estimators_keta.condition == 'keta')&(df_estimators_keta.decay_isol > 1/30.)

sns.lineplot(x='time', y='decay_isol_mod', markers='o',lw=3,estimator=np.median,ci=95, color=color_mmf,data=df_estimators_mmf.dropna(),label='MMF')
sns.lineplot(x='time', y='decay_isol_mod', markers='o',lw=3,estimator=np.median,ci=95, color=color_keta,data=df_estimators_keta.dropna(),label='Keta/Xyl')
sns.lineplot(x='time', y='decay_isol_mod', markers='o',lw=3,estimator=np.median,ci=95, color=color_iso,data=df_estimators_iso.dropna(),label='Iso')


plt.xlim([0,360])
plt.legend(fontsize=15)
#plt.title("relative decay constant",fontsize=22)
plt.ylabel("modulation index (decay)",fontsize=15)
plt.xlabel("time (min)", fontsize=15)
#plt.ylim([0.5,1.5])

plt.savefig("./decay_isol_mod.svg", format = 'svg', dpi=300)
#plt.savefig('height.median_norm.eps', format='eps')
plt.savefig('./decay_isol_mod.png', format='png')


#################### SAVE DATA FOR STATS ########################
#################################################################

df_estimators_mmf['animal'] = df_estimators_mmf['animal'].apply(lambda x: x.split('TS_C_MMF_')[1])
df_estimators_keta['animal'] = df_estimators_keta['animal'].apply(lambda x: x.split('TS_C_KETA_')[1])
df_estimators_iso['animal'] = df_estimators_iso['animal'].apply(lambda x: x.split('TS_C_ISO_')[1])


df_estimators_mmf['condition'] = 'mmf'
df_estimators_keta['condition'] = 'keta'
df_estimators_iso['condition'] = 'iso'

df_estimators = pd.concat([df_estimators_mmf,df_estimators_keta,df_estimators_iso])

df_estimators[['time','condition','animal','neuronID','n_peaks_mod','height.median_mod','decay_isol_mod']].to_excel("/media/andrey/My Passport/Yang_Chini_et_al/Stats (R)/scripts/transition_state_modulation_index.xlsx")



