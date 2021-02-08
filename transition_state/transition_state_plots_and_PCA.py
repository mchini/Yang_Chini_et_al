
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

df_estimators_old_mmf = pd.read_pickle(data_path + "df_estimators_TS_C_MMF_F0_M0_n_peaks_baseline.pkl")
df_estimators_new_mmf = pd.read_pickle(data_path + "df_estimators_TS_C_MMF_8235_8237_8238_n_peaks_baseline.pkl")
df_estimators_mmf = pd.concat([df_estimators_new_mmf,df_estimators_old_mmf])

df_estimators_new_iso = pd.read_pickle(data_path + "df_estimators_TS_C_ISO_8235_8237_8238_n_peaks_baseline.pkl")
df_estimators_old_iso = pd.read_pickle(data_path + "df_estimators_TS_C_ISO_F0_M0_n_peaks_baseline.pkl")
df_estimators_iso = pd.concat([df_estimators_new_iso,df_estimators_old_iso])

df_estimators_new_keta = pd.read_pickle(data_path + "df_estimators_TS_C_KETA_8235_8237_8238_n_peaks_baseline.pkl")
df_estimators_old_keta = pd.read_pickle(data_path + "df_estimators_TS_C_KETA_F0_M3_n_peaks_baseline.pkl")
df_estimators_keta = pd.concat([df_estimators_new_keta,df_estimators_old_keta])


df_estimators_awake = pd.read_pickle(data_path + "df_estimators_TS_C_AWAKE_F0_8235_8237_8238_n_peaks_baseline.pkl")




## TODO CHECK TIMING

condition_to_time_mmf = {"awake":0,"mmf":30,"post30":60,"post60":90,"post120":150,"post180":210,"post240":270,"post300":330,"post360":390}
condition_to_time_keta = {"awake":0,"keta":30,"keta30":60,"keta60":90,"keta120":150,"keta180":210,"keta240":270,"keta300":330,"keta360":390}
condition_to_time_iso = {"awake":0,"iso":30,"rec30":60,"rec60":90,"rec120":150,"rec180":210,"rec240":270,"rec300":330,"rec360":390}
condition_to_time_awake = {"awake1":0,"awake2":60,"awake3":120,"awake4":180,"awake5":240,"awake6":300,"awake7":360,"awake8":420}  # 420 is only availible for F0 animal

df_estimators_mmf['time'] = [condition_to_time_mmf[x] for x in df_estimators_mmf['condition']]
df_estimators_keta['time'] = [condition_to_time_keta[x] for x in df_estimators_keta['condition']]
df_estimators_iso['time'] = [condition_to_time_iso[x] for x in df_estimators_iso['condition']]

df_estimators_awake['time'] = [condition_to_time_awake[x] for x in df_estimators_awake['condition']]





df_corrs_old_mmf = pd.read_pickle(data_path + "df_corrs_TS_C_MMF_F0_M0_n_peaks_baseline.pkl")
df_corrs_new_mmf = pd.read_pickle(data_path + "df_corrs_TS_C_MMF_8235_8237_8238_n_peaks_baseline.pkl")
df_corrs_mmf = pd.concat([df_corrs_new_mmf,df_corrs_old_mmf])

df_corrs_new_iso = pd.read_pickle(data_path + "df_corrs_TS_C_ISO_8235_8237_8238_n_peaks_baseline.pkl")
df_corrs_old_iso = pd.read_pickle(data_path + "df_corrs_TS_C_ISO_F0_M0_n_peaks_baseline.pkl")
df_corrs_iso = pd.concat([df_corrs_new_iso,df_corrs_old_iso])

df_corrs_new_keta = pd.read_pickle(data_path + "df_corrs_TS_C_KETA_8235_8237_8238_n_peaks_baseline.pkl")
df_corrs_old_keta = pd.read_pickle(data_path + "df_corrs_TS_C_KETA_F0_M3_n_peaks_baseline.pkl")
df_corrs_keta = pd.concat([df_corrs_new_keta,df_corrs_old_keta])


df_corrs_awake = pd.read_pickle(data_path + "df_corrs_TS_C_AWAKE_F0_8235_8237_8238_n_peaks_baseline.pkl")

color_awake = (0,191/255,255/255)
color_ctrl = (0,0/255,255/255)
color_mmf = (245/255,143/255,32/255)
color_keta = (181./255,34./255,48./255)
color_iso = (143./255,39./255,143./255)

#n_peaks_mmf =  df_estimators_mmf[(df_estimators_mmf.animal=='TS_C_MMF_F0')].groupby('condition', as_index=False)['n_peaks']

#n_peaks_mmf_std =  df_estimators_mmf[(df_estimators_mmf.animal=='TS_C_MMF_F0')].groupby('condition', as_index=False)['n_peaks'].std()

##sns.pointplot(x='time', y='n_peaks', markers='o', scale=1.1, ci =95,err_style="band", color='C1',data=df_estimators_mmf, order =["awake","mmf","post30","post60","post120","post180","post240","post300","post360"])

###################################################################################################
################ VIOLIN PLOTS #####################################################################
###################################################################################################

################################## N_PEAKS #######################################################

#sel = (df_estimators_conditions.animal=='TS_C_ISO_F0')|(df_estimators_conditions.animal=='TS_C_ISO_F1')

# N_PEAKS ISO

param = 'n_peaks'

ymin = 1
ymax = 150

f, axes = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
sns.despine(left=True)

axes.set_ylim([ymin, ymax])
axes.set_yscale('log')

sel = (df_estimators_iso.n_peaks > 0.3)  ### Exclude zero peaks
    

sns.violinplot(x='condition', y=param, data=df_estimators_iso[sel],ax=axes, order = ["awake","iso","rec30","rec60","rec120","rec180","rec240","rec300"])
sns.pointplot(x='condition', y=param, data=df_estimators_iso[sel].groupby('condition', as_index=False).median(), ax=axes, markers='o', order = ["awake","iso","rec30","rec60","rec120","rec180","rec240","rec300"],color='k')

plt.savefig("n_peaks_iso.svg", format = 'svg', dpi=300)
plt.savefig('n_peaks_iso.png', format='png')

################################################################################################

# N_PEAKS KETA

param = 'n_peaks'

ymin = 1
ymax = 150

f, axes = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
sns.despine(left=True)

axes.set_ylim([ymin, ymax])
axes.set_yscale('log')

sel = (df_estimators_keta.n_peaks > 0.3) ### Exclude zero peaks


sns.violinplot(x='condition', y=param, data=df_estimators_keta[sel],ax=axes, order = ["awake","keta","keta30","keta60","keta120","keta180","keta240","keta300"])
sns.pointplot(x='condition', y=param, data=df_estimators_keta[sel].groupby('condition', as_index=False).median(), ax=axes, markers='o', order = ["awake","keta","keta30","keta60","keta120","keta180","keta240","keta300"],color='k')

plt.savefig("n_peaks_keta.svg", format = 'svg', dpi=300)
plt.savefig('n_peaks_keta.png', format='png')

############################################################################################

# N_PEAKS MMF

param = 'n_peaks'

ymin = 1
ymax = 150

f, axes = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
sns.despine(left=True)

axes.set_ylim([ymin, ymax])
axes.set_yscale('log')

sel = (df_estimators_mmf.n_peaks > 0.3) ### Exclude zero peaks

sns.violinplot(x='condition', y=param, data=df_estimators_mmf[sel],ax=axes, order = ["awake","mmf","post30","post60","post120","post180","post240","post300"])
sns.pointplot(x='condition', y=param, data=df_estimators_mmf[sel].groupby('condition', as_index=False).median(), ax=axes,  markers='o', order = ["awake","mmf","post30","post60","post120","post180","post240","post300"],color='k')

plt.savefig("n_peaks_mmf.svg", format = 'svg', dpi=300)
plt.savefig('n_peaks_mmf.png', format='png')

###########################################################################################
    
# N_PEAKS AWAKE

param = 'n_peaks'

ymin = 1
ymax = 150

f, axes = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
sns.despine(left=True)

axes.set_ylim([ymin, ymax])
axes.set_yscale('log')

sel = (df_estimators_awake.n_peaks > 0.3) ### Exclude zero peaks

sns.violinplot(x='condition', y=param, data=df_estimators_awake[sel],ax=axes, order = ["awake1","awake2","awake3","awake4","awake5","awake6","awake7"])
sns.pointplot(x='condition', y=param, data=df_estimators_awake[sel].groupby('condition', as_index=False).median(), ax=axes, markers='o', order = ["awake1","awake2","awake3","awake4","awake5","awake6","awake7"],color='k')

plt.savefig("n_peaks_awake.svg", format = 'svg', dpi=300)
plt.savefig('n_peaks_awake.png', format='png')


###########################################################################################
###########################################################################################
################################## HEIGHT.MEDIAN ##########################################

# HEIGHT.MEDIAN ISO

param = 'height.median'

ymin = 50
ymax = 3000

f, axes = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
sns.despine(left=True)

axes.set_ylim([ymin, ymax])
axes.set_yscale('log')

sel = (df_estimators_iso.n_peaks > 0.3)  ### Exclude zero peaks


sns.violinplot(x='condition', y=param, data=df_estimators_iso[sel],ax=axes,  order = ["awake","iso","rec30","rec60","rec120","rec180","rec240","rec300"])

sns.pointplot(x='condition', y=param, data=df_estimators_iso[sel].groupby('condition', as_index=False).median(),  color='k', ax=axes, order = ["awake","iso","rec30","rec60","rec120","rec180","rec240","rec300"])

plt.savefig("height.median_iso.svg", format = 'svg', dpi=300)
plt.savefig('height.median_iso.png', format='png')

# HEIGHT.MEDIAN KETA

param = 'height.median'

ymin = 50
ymax = 3000

f, axes = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
sns.despine(left=True)

axes.set_ylim([ymin, ymax])
axes.set_yscale('log')

sel = (df_estimators_keta.n_peaks > 0.3)


#animals_select = (df_estimators_keta.animal != 'TS_C_KETA_F0')&(df_estimators_keta.animal != 'TS_C_KETA_M3')
    
sns.pointplot(x='condition', y=param, data=df_estimators_keta[sel].groupby('condition', as_index=False).median(), ax=axes, order = ["awake","keta","keta30","keta60","keta120","keta180","keta240","keta300"], color='k')
sns.violinplot(x='condition', y=param, data=df_estimators_keta[sel],ax=axes, order = ["awake","keta","keta30","keta60","keta120","keta180","keta240","keta300"])

plt.savefig("height.median_keta.svg", format = 'svg', dpi=300)
plt.savefig('height.median_keta.png', format='png')


# HEIGHT.MEDIAN MMF

param = 'height.median'

ymin = 50
ymax = 3000

f, axes = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
sns.despine(left=True)

axes.set_ylim([ymin, ymax])
axes.set_yscale('log')

sel = (df_estimators_mmf.n_peaks > 0.3)


sns.violinplot(x='condition', y=param, data=df_estimators_mmf[sel],ax=axes, order = ["awake","mmf","post30","post60","post120","post180","post240","post300"])
sns.pointplot(x='condition', y=param, data=df_estimators_mmf[sel].groupby('condition',  as_index=False).median(), ax=axes, order = ["awake","mmf","post30","post60","post120","post180","post240","post300"], color='k')

plt.savefig("height.median_mmf.svg", format = 'svg', dpi=300)
plt.savefig('height.median_mmf.png', format='png')

# HEIGHT.MEDIAN AWAKE

param = 'height.median'

ymin = 50
ymax = 3000

f, axes = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
sns.despine(left=True)

#axes.set_ylim([ymin, ymax])
axes.set_yscale('log')

sel = (df_estimators_awake.n_peaks > 0.3)


sns.violinplot(x='condition', y=param, data=df_estimators_awake[sel],ax=axes, order = ["awake1","awake2","awake3","awake4","awake5","awake6","awake7"])
sns.pointplot(x='condition', y=param, data=df_estimators_awake[sel].groupby('condition', as_index=False).median(), ax=axes, order = ["awake1","awake2","awake3","awake4","awake5","awake6","awake7"],color='k')

plt.savefig("height.median_awake.svg", format = 'svg', dpi=300)
plt.savefig('height.median_awake.png', format='png')


####################################################################################
####################################################################################
################################## decay_isol ######################################

# decay_isol ISO

param = 'decay_isol'
ymin = 0.0
ymax = 1.2

f, axes = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
sns.despine(left=True)

axes.set_ylim([ymin, ymax])
#xes.set_yscale('log')


sel = (df_estimators_iso.decay_isol < 10)&(df_estimators_iso.n_peaks > 0.3)
    

sns.violinplot(x='condition', y=param, data=df_estimators_iso[sel],ax=axes, order = ["awake","iso","rec30","rec60","rec120","rec180","rec240","rec300"])
sns.pointplot(x='condition', y=param, data=df_estimators_iso[sel].groupby('condition', as_index=False).median(), ax=axes, order = ["awake","iso","rec30","rec60","rec120","rec180","rec240","rec300"], color='k')

plt.savefig("decay_isol_iso.svg", format = 'svg', dpi=300)
plt.savefig('decay_isol_iso.png', format='png')

# decay_isol MMF

param = 'decay_isol'
ymin = 0.0
ymax = 1.2

f, axes = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
sns.despine(left=True)

axes.set_ylim([ymin, ymax])
#axes.set_yscale('log')

if param != 'decay_isol':
    sel = True
else:
    sel = (df_estimators_mmf.decay_isol < 10)&(df_estimators_mmf.n_peaks > 0.3)


sns.violinplot(x='condition', y=param, data=df_estimators_mmf[sel],ax=axes, order = ["awake","mmf","post30","post60","post120","post180","post240","post300"])
sns.pointplot(x='condition', y=param, data=df_estimators_mmf[sel].groupby('condition', as_index=False).median(), ax=axes, order = ["awake","mmf","post30","post60","post120","post180","post240","post300"],color='k')

plt.savefig("decay_isol_mmf.svg", format = 'svg', dpi=300)
plt.savefig('decay_isol_mmf.png', format='png')

# decay_isol KETA

param = 'decay_isol'
ymin = 0.0
ymax = 1.2

f, axes = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
sns.despine(left=True)

axes.set_ylim([ymin, ymax])
#axes.set_yscale('log')

sel = (df_estimators_keta.decay_isol < 10)&(df_estimators_keta.n_peaks > 0.3)


sns.violinplot(x='condition', y=param, data=df_estimators_keta[sel],ax=axes, order = ["awake","keta","keta30","keta60","keta120","keta180","keta240","keta300"])
sns.pointplot(x='condition', y=param, data=df_estimators_keta[sel].groupby('condition', as_index=False).median(), ax=axes, order = ["awake","keta","keta30","keta60","keta120","keta180","keta240","keta300"],color='k')

plt.savefig("decay_isol_keta.svg", format = 'svg', dpi=300)
plt.savefig('decay_isol_keta.png', format='png')

# decay_isol AWAKE

param = 'decay_isol'
ymin = 0.0
ymax = 1.2

f, axes = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
sns.despine(left=True)

axes.set_ylim([ymin, ymax])
#axes.set_yscale('log')

#if param != 'decay_isol':
#    sel = True
#else:
sel = (df_estimators_awake.decay_isol < 10)&(df_estimators_awake.n_peaks > 0.3)


sns.violinplot(x='condition', y=param, data=df_estimators_awake[sel],ax=axes, order = ["awake1","awake2","awake3","awake4","awake5","awake6","awake7"])
sns.pointplot(x='condition', y=param, data=df_estimators_awake[sel].groupby('condition', as_index=False).median(), ax=axes, order = ["awake1","awake2","awake3","awake4","awake5","awake6","awake7"],color='k')

plt.savefig("decay_isol_awake.svg", format = 'svg', dpi=300)
plt.savefig('decay_isol_awake.png', format='png')


###############################################################################################################
###############################################################################################################
################################################### CORRELATIONS ##############################################

##########  CORR ISO ###############

param = 'Tm0p7Ncorr.abs'
ymin = 0.0
ymax = 0.05

f, axes = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
sns.despine(left=True)

axes.set_ylim([ymin, ymax])
axes.set_yscale('log')

#df_corrs_iso_test = df_corrs_iso[0:10000]

data=df_corrs_iso.fillna(0, inplace=True) 
    
sns.pointplot(x='condition', y=param, data=df_corrs_iso.groupby('condition', as_index=False).median(), ax=axes, color='k', order = ["awake","iso","rec30","rec60","rec120","rec180","rec240","rec300"])
sns.violinplot(x='condition', y=param, data=df_corrs_iso,ax=axes, order = ["awake","iso","rec30","rec60","rec120","rec180","rec240","rec300"])


plt.savefig("Tm0p7Ncorr_iso_log.svg", format = 'svg', dpi=300)
plt.savefig("Tm0p7Ncorr_iso_log.png", format='png')



######### CORR KETA #################

param = 'Tm0p7Ncorr.abs'
ymin = 0.0
ymax = 0.05


f, axes = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
sns.despine(left=True)

axes.set_ylim([ymin, ymax])
axes.set_yscale('log')
  
sns.pointplot(x='condition', y=param, data=df_corrs_keta.groupby('condition', as_index=False).median(), ax=axes, color='k',  order = ["awake","keta","keta30","keta60","keta120","keta180","keta240","keta300"])
sns.violinplot(x='condition', y=param, data=df_corrs_keta,ax=axes, order = ["awake","keta","keta30","keta60","keta120","keta180","keta240","keta300"])



plt.savefig("Tm0p7Ncorr_keta_log.svg", format = 'svg', dpi=300)
plt.savefig("Tm0p7Ncorr_keta_log.png", format='png')



######### CORR MMF #################

param = 'Tm0p7Ncorr.abs'
ymin = 0.0
ymax = 0.05

f, axes = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
sns.despine(left=True)

axes.set_ylim([ymin, ymax])
axes.set_yscale('log')
  
sns.pointplot(x='condition', y=param, data=df_corrs_mmf.groupby('condition', as_index=False).median(), ax=axes, color='k', order = ["awake","mmf","post30","post60","post120","post180","post240","post300"])
sns.violinplot(x='condition', y=param, data=df_corrs_mmf,ax=axes, order = ["awake","mmf","post30","post60","post120","post180","post240","post300"])



plt.savefig("Tm0p7Ncorr_mmf_log.svg", format = 'svg', dpi=300)
plt.savefig("Tm0p7Ncorr_mmf_log.png", format='png')



######### CORR AWAKE ##############

param = 'Tm0p7Ncorr.abs'
ymin = 0.0
ymax = 0.05

f, axes = plt.subplots(1, 1, figsize=(8, 6), sharex=True)
sns.despine(left=True)

axes.set_ylim([ymin, ymax])
axes.set_yscale('log')
    
sns.pointplot(x='condition', y=param, data=df_corrs_awake.groupby('condition', as_index=False).median(), ax=axes, color='k', order = ["awake1","awake2","awake3","awake4","awake5","awake6","awake7"])
sns.violinplot(x='condition', y=param, data=df_corrs_awake,ax=axes, order = ["awake1","awake2","awake3","awake4","awake5","awake6","awake7"])


plt.savefig("Tm0p7Ncorr_awake_log.svg", format = 'svg', dpi=300)
plt.savefig("Tm0p7Ncorr_awake_log.png", format='png')



###############################################################################################################
###############################################################################################################


#sns.lineplot(x='condition', y='n_peaks', markers='o', ci =95,estimator=np.median,err_style="band", color='C1',data=df_estimators_mmf,label='MMF')


sns.lineplot(x='time', y='n_peaks', markers='o', ci =95,err_style="band", color='C1',data=df_estimators_mmf,label='MMF')
sns.lineplot(x='time', y='n_peaks', markers='o', ci =95,err_style="band", color='C2',data=df_estimators_iso,label='ISO')
sns.lineplot(x='time', y='n_peaks', markers='o', ci =95,err_style="band", color='C3',data=df_estimators_keta,label='Keta/Xyl')
sns.lineplot(x='time', y='n_peaks', markers='o', ci =95,err_style="band", color='C0',data=df_estimators_awake,label='Awake')

sns.lineplot(x='time', y='height.median', markers='o', estimator=np.median,ci =95,err_style="band", color='C1',data=df_estimators_mmf,label='MMF')
sns.lineplot(x='time', y='height.median', markers='o', estimator=np.median,ci =95,err_style="band", color='C2',data=df_estimators_iso,label='ISO')
sns.lineplot(x='time', y='height.median', markers='o', estimator=np.median,ci =95,err_style="band", color='C3',data=df_estimators_keta,label='Keta/Xyl')
sns.lineplot(x='time', y='height.median', markers='o', estimator=np.median,ci =95,err_style="band", color='C0',data=df_estimators_awake,label='Awake')
plt.xrange([0.0, 360.0])
plt.show()
#sns.pointplot(x='time', y='n_peaks', markers='o', scale=1.1, color='k',data=df_estimators_mmf)

### CALCULATE LOG-NORMALIZED N_PEAKS MMF

df_estimators_mmf['n_peaks_norm'] = df_estimators_mmf['n_peaks']

sel = (df_estimators_mmf.n_peaks > 0.3)

#mmf_awake_median = df_estimators_mmf[sel&(df_estimators_mmf.condition == 'awake')]['n_peaks'].median()

for animal in ['TS_C_MMF_8235', 'TS_C_MMF_8237', 'TS_C_MMF_8238', 'TS_C_MMF_F0','TS_C_MMF_M0']: 
    
    # Normalize with median of all animals in awake condtion START    
    #xxx = df_estimators_mmf.loc[sel&(df_estimators_mmf.animal == animal), 'n_peaks_norm' ] 
    #xxx /=  mmf_awake_median
    #xxx = np.log10(xxx)
    #df_estimators_mmf.loc[sel&(df_estimators_mmf.animal == animal), 'n_peaks_norm' ] = xxx
    #df_estimators_mmf.loc[~(sel)&(df_estimators_mmf.animal == animal), 'n_peaks_norm' ] = np.nan  
    # Normalize with median of all animals in awake condtion END

        
    # Normalize with median of each individual animal in awake condtion START    
    sel_animal = (df_estimators_mmf.animal == animal)
    mmf_awake_median = df_estimators_mmf[sel_animal&sel&(df_estimators_mmf.condition == 'awake')]['n_peaks'].median()
    yyy = df_estimators_mmf.loc[sel_animal&sel, 'n_peaks_norm' ]
    yyy /=  mmf_awake_median
    yyy = np.log10(yyy)
    df_estimators_mmf.loc[sel_animal&sel, 'n_peaks_norm']  = yyy
    df_estimators_mmf.loc[~sel&sel_animal, 'n_peaks_norm' ] = np.nan
    # Normalize with median of each individual animal in awake condtion END  


df_estimators_iso['n_peaks_norm'] = df_estimators_iso['n_peaks']

sel = (df_estimators_iso.n_peaks > 0.3)

#iso_awake_median = df_estimators_iso[sel&(df_estimators_iso.condition == 'awake')]['n_peaks'].median()

for animal in ['TS_C_ISO_8235', 'TS_C_ISO_8237', 'TS_C_ISO_8238', 'TS_C_ISO_M0','TS_C_ISO_F0']:   
    
    # Normalize with median of all animals in awake condtion START 
    #xxx =  df_estimators_iso.loc[sel&(df_estimators_iso.animal == animal), 'n_peaks_norm' ]
    #xxx /=  iso_awake_median
    #xxx = np.log10(xxx)
    #df_estimators_iso.loc[sel&(df_estimators_iso.animal == animal), 'n_peaks_norm' ] = xxx
    #df_estimators_iso.loc[~(sel)&(df_estimators_iso.animal == animal), 'n_peaks_norm' ] = np.nan
    # Normalize with median of all animals in awake condtion END
    
    # Normalize with median of each individual animal in awake condtion START    
    sel_animal = (df_estimators_iso.animal == animal)
    iso_awake_median = df_estimators_iso[sel_animal&sel&(df_estimators_iso.condition == 'awake')]['n_peaks'].median()
    yyy = df_estimators_iso.loc[sel_animal&sel, 'n_peaks_norm' ]
    yyy /=  iso_awake_median
    yyy = np.log10(yyy)
    df_estimators_iso.loc[sel_animal&sel, 'n_peaks_norm']  = yyy
    df_estimators_iso.loc[~sel&sel_animal, 'n_peaks_norm' ] = np.nan
    # Normalize with median of each individual animal in awake condtion END  
    
    

df_estimators_keta['n_peaks_norm'] = df_estimators_keta['n_peaks']

sel = (df_estimators_keta.n_peaks > 0.3)

keta_awake_median = df_estimators_keta[sel&(df_estimators_keta.condition == 'awake')]['n_peaks'].median()

for animal in ['TS_C_KETA_8235', 'TS_C_KETA_8237', 'TS_C_KETA_8238', 'TS_C_KETA_F0','TS_C_KETA_M3']:
    
    # Normalize with median of all animals in awake condtion START     
    #xxx =  df_estimators_keta.loc[sel&(df_estimators_keta.animal == animal), 'n_peaks_norm' ] 
    #xxx /=  keta_awake_median
    #xxx = np.log10(xxx)
    #df_estimators_keta.loc[sel&(df_estimators_keta.animal == animal), 'n_peaks_norm' ]  = xxx
    #df_estimators_keta.loc[~(sel)&(df_estimators_keta.animal == animal), 'n_peaks_norm' ]  = np.nan
    # Normalize with median of all animals in awake condtion END    
    
    # Normalize with median of each individual animal in awake condtion START    
    sel_animal = (df_estimators_keta.animal == animal)
    keta_awake_median = df_estimators_keta[sel_animal&sel&(df_estimators_keta.condition == 'awake')]['n_peaks'].median()
    yyy = df_estimators_keta.loc[sel_animal&sel, 'n_peaks_norm' ]
    yyy /=  keta_awake_median
    yyy = np.log10(yyy)
    df_estimators_keta.loc[sel_animal&sel, 'n_peaks_norm']  = yyy
    df_estimators_keta.loc[~sel&sel_animal, 'n_peaks_norm' ] = np.nan
    # Normalize with median of each individual animal in awake condtion END     
    
df_estimators_awake['n_peaks_norm'] = df_estimators_awake['n_peaks']

sel = (df_estimators_awake.n_peaks > 0.3)

#awake1_median = df_estimators_awake[sel&(df_estimators_awake.condition == 'awake1')]['n_peaks'].median()

for animal in ['TS_C_AWAKE_CTRL_8235', 'TS_C_AWAKE_CTRL_8237', 'TS_C_AWAKE_CTRL_8238', 'TS_C_AWAKE_CTRL_F0']:
    
     # Normalize with median of all animals in awake condtion START    
    #xxx =  df_estimators_awake.loc[sel&(df_estimators_awake.animal == animal), 'n_peaks_norm' ] 
    #xxx /=  awake1_median 
    #xxx = np.log10(xxx)
    #df_estimators_awake.loc[sel&(df_estimators_awake.animal == animal), 'n_peaks_norm' ]  = xxx
    #df_estimators_awake.loc[~(sel)&(df_estimators_awake.animal == animal), 'n_peaks_norm' ]  = np.nan
    # Normalize with median of all animals in awake condtion END   
    
    # Normalize with median of each individual animal in awake condtion START    
    sel_animal = (df_estimators_awake.animal == animal)
    awake1_median = df_estimators_awake[sel_animal&sel&(df_estimators_awake.condition == 'awake1')]['n_peaks'].median()
    yyy = df_estimators_awake.loc[sel_animal&sel, 'n_peaks_norm' ]
    yyy /=  awake1_median
    yyy = np.log10(yyy)
    df_estimators_awake.loc[sel_animal&sel, 'n_peaks_norm']  = yyy
    df_estimators_awake.loc[~sel&sel_animal, 'n_peaks_norm' ] = np.nan
    # Normalize with median of each individual animal in awake condtion END      
    


################################################################
################################################################

#### AWAKE: normalization by global median

#df_estimators_awake['n_peaks_norm'] = df_estimators_awake['n_peaks']
#for animal in ['TS_C_AWAKE_CTRL_8235', 'TS_C_AWAKE_CTRL_8237', 'TS_C_AWAKE_CTRL_8238']:
 #   sel = (df_estimators_awake.animal == animal)
#    awake1_median = df_estimators_awake[sel&(df_estimators_awake.condition == 'awake1')]['n_peaks'].median()
#    df_estimators_awake.loc[df_estimators_awake.animal == animal, 'n_peaks_norm' ] /=  awake1_median


### CALCULATE NORMALIZED AMPLITUDE
    
df_estimators_mmf['height.median_norm'] = df_estimators_mmf['height.median']

sel = (df_estimators_mmf.n_peaks > 0.3)

#mmf_awake_median = df_estimators_mmf[sel&(df_estimators_mmf.condition == 'awake')]['height.median'].median()

for animal in ['TS_C_MMF_8235', 'TS_C_MMF_8237', 'TS_C_MMF_8238','TS_C_MMF_F0','TS_C_MMF_M0']:  #  


    # Normalize with median of all animals in awake condtion START
    #xxx = df_estimators_mmf.loc[sel&(df_estimators_mmf.animal == animal), 'height.median_norm' ] 
    #xxx /=   mmf_awake_median 
    #xxx = np.log10(xxx)
    #df_estimators_mmf.loc[sel&(df_estimators_mmf.animal == animal), 'height.median_norm' ]  = xxx
    #df_estimators_mmf.loc[~sel&(df_estimators_mmf.animal == animal), 'height.median_norm' ] = np.nan
    # Normalize with median of all animals in awake condtion END
    
        
    # Normalize with median of each individual animal in awake condtion START    
    sel_animal = (df_estimators_mmf.animal == animal)
    mmf_awake_median = df_estimators_mmf[sel_animal&sel&(df_estimators_mmf.condition == 'awake')]['height.median'].median()
    yyy = df_estimators_mmf.loc[sel_animal&sel, 'height.median_norm' ]
    yyy /=  mmf_awake_median
    yyy = np.log10(yyy)
    df_estimators_mmf.loc[sel_animal&sel, 'height.median_norm' ]  = yyy
    df_estimators_mmf.loc[~sel&sel_animal, 'height.median_norm' ] = np.nan
    # Normalize with median of each individual animal in awake condtion END  
    



df_estimators_iso['height.median_norm'] = df_estimators_iso['height.median']

sel = (df_estimators_iso.n_peaks > 0.3)

#iso_awake_median = df_estimators_iso[sel&(df_estimators_iso.condition == 'awake')]['height.median'].median()

for animal in ['TS_C_ISO_8235', 'TS_C_ISO_8237', 'TS_C_ISO_8238', 'TS_C_ISO_M0', 'TS_C_ISO_F0']: 
    
    # Normalize with median of all animals in awake condtion START
    #xxx = df_estimators_iso.loc[sel&(df_estimators_iso.animal == animal), 'height.median_norm' ]
    #xxx /=   iso_awake_median
    #xxx = np.log10(xxx)
    #df_estimators_iso.loc[sel&(df_estimators_iso.animal == animal), 'height.median_norm' ]  = xxx
    #df_estimators_iso.loc[~sel&(df_estimators_iso.animal == animal), 'height.median_norm' ] = np.nan   
    # Normalize with median of all animals in awake condtion END
    
    
    # Normalize with median of each individual animal in awake condtion START    
    sel_animal = (df_estimators_iso.animal == animal)
    iso_awake_median = df_estimators_iso[sel_animal&sel&(df_estimators_iso.condition == 'awake')]['height.median'].median()
    yyy = df_estimators_iso.loc[sel_animal&sel, 'height.median_norm' ]
    yyy /=  iso_awake_median
    yyy = np.log10(yyy)
    df_estimators_iso.loc[sel_animal&sel, 'height.median_norm' ]  = yyy
    df_estimators_iso.loc[~sel&sel_animal, 'height.median_norm' ] = np.nan
    # Normalize with median of each individual animal in awake condtion END     






df_estimators_keta['height.median_norm'] = df_estimators_keta['height.median']

sel = (df_estimators_keta.n_peaks > 0.3)

#   keta_awake_median = df_estimators_keta[sel&(df_estimators_keta.condition == 'awake')]['height.median'].median()

for animal in ['TS_C_KETA_8235', 'TS_C_KETA_8237', 'TS_C_KETA_8238', 'TS_C_KETA_F0','TS_C_KETA_M3']:  
    
    # Normalize with median of all animals in awake condtion START
    #xxx = df_estimators_keta.loc[sel&(df_estimators_keta.animal == animal), 'height.median_norm' ]
    #xxx /=   keta_awake_median
    #xxx = np.log10(xxx)
    #df_estimators_keta.loc[sel&(df_estimators_keta.animal == animal), 'height.median_norm' ]  = xxx
    #df_estimators_keta.loc[~sel&(df_estimators_keta.animal == animal), 'height.median_norm' ] = np.nan
    # Normalize with median of all animals in awake condtion END
    
    # Normalize with median of each individual animal in awake condtion START    
    sel_animal = (df_estimators_keta.animal == animal)
    keta_awake_median = df_estimators_keta[sel_animal&sel&(df_estimators_keta.condition == 'awake')]['height.median'].median()
    yyy = df_estimators_keta.loc[sel_animal&sel, 'height.median_norm' ]
    yyy /=  keta_awake_median
    yyy = np.log10(yyy)
    df_estimators_keta.loc[sel_animal&sel, 'height.median_norm' ]  = yyy
    df_estimators_keta.loc[~sel&sel_animal, 'height.median_norm' ] = np.nan
    # Normalize with median of each individual animal in awake condtion END  

    
df_estimators_awake['height.median_norm'] = df_estimators_awake['height.median']

sel = (df_estimators_awake.n_peaks > 0.3)

#awake1_median = df_estimators_awake[sel&(df_estimators_awake.condition == 'awake1')]['height.median'].median()

for animal in ['TS_C_AWAKE_CTRL_8235', 'TS_C_AWAKE_CTRL_8237', 'TS_C_AWAKE_CTRL_8238', 'TS_C_AWAKE_CTRL_F0']:
     # Normalize with median of all animals in awake condtion START
    #xxx = df_estimators_awake.loc[sel&(df_estimators_awake.animal == animal), 'height.median_norm' ]
    #xxx /=   awake1_median
    #xxx = np.log10(xxx)
    #df_estimators_awake.loc[sel&(df_estimators_awake.animal == animal), 'height.median_norm' ]  = xxx
    #df_estimators_awake.loc[~sel&(df_estimators_awake.animal == animal), 'height.median_norm' ] = np.nan 
     # Normalize with median of all animals in awake condtion END
    
    # Normalize with median of each individual animal in awake condtion START    
    sel_animal = (df_estimators_awake.animal == animal)
    awake1_median = df_estimators_awake[sel_animal&sel&(df_estimators_awake.condition == 'awake1')]['height.median'].median()
    yyy = df_estimators_awake.loc[sel_animal&sel, 'height.median_norm' ]
    yyy /=  awake1_median
    yyy = np.log10(yyy)
    df_estimators_awake.loc[sel_animal&sel, 'height.median_norm' ]  = yyy
    df_estimators_awake.loc[~sel&sel_animal, 'height.median_norm' ] = np.nan
    # Normalize with median of each individual animal in awake condtion END 
 
    
### 
    
    
### CALCULATE NORMALIZED DECAY CONSTANT



df_estimators_mmf['decay_isol_norm'] = df_estimators_mmf['decay_isol']

sel = (df_estimators_mmf.n_peaks > 0.3)

#mmf_awake_median = df_estimators_mmf[sel&(df_estimators_mmf.condition == 'awake')]['decay_isol'].median()

for animal in ['TS_C_MMF_8235', 'TS_C_MMF_8237', 'TS_C_MMF_8238','TS_C_MMF_F0','TS_C_MMF_M0']:  #  
    sel_animal = (df_estimators_mmf.animal == animal)
    mmf_awake_median = df_estimators_mmf[sel&sel_animal&(df_estimators_mmf.condition == 'awake')]['decay_isol'].median()
    df_estimators_mmf.loc[sel&sel_animal, 'decay_isol_norm' ] /=  mmf_awake_median


df_estimators_iso['decay_isol_norm'] = df_estimators_iso['decay_isol']

sel = (df_estimators_iso.n_peaks > 0.3)

#iso_awake_median = df_estimators_iso[sel&(df_estimators_iso.condition == 'awake')]['decay_isol'].median()

for animal in ['TS_C_ISO_8235', 'TS_C_ISO_8237', 'TS_C_ISO_8238', 'TS_C_ISO_M0', 'TS_C_ISO_F0']:   
    sel_animal = (df_estimators_iso.animal == animal)
    iso_awake_median = df_estimators_iso[sel&sel_animal&(df_estimators_iso.condition == 'awake')]['decay_isol'].median()
    df_estimators_iso.loc[sel&sel_animal, 'decay_isol_norm' ] /=  iso_awake_median
    

df_estimators_keta['decay_isol_norm'] = df_estimators_keta['decay_isol']

sel = (df_estimators_keta.n_peaks > 0.3)

#keta_awake_median = df_estimators_keta[sel&(df_estimators_keta.condition == 'awake')]['decay_isol'].median()

for animal in ['TS_C_KETA_8235', 'TS_C_KETA_8237', 'TS_C_KETA_8238', 'TS_C_KETA_F0','TS_C_KETA_M3']:  
    sel_animal = (df_estimators_keta.animal == animal)
    keta_awake_median = df_estimators_keta[sel&sel_animal&(df_estimators_keta.condition == 'awake')]['decay_isol'].median()
    df_estimators_keta.loc[sel&sel_animal, 'decay_isol_norm' ] /=  keta_awake_median
    
    
df_estimators_awake['decay_isol_norm'] = df_estimators_awake['decay_isol']

sel = (df_estimators_awake.n_peaks > 0.3)

#awake1_median =df_estimators_awake[sel&(df_estimators_awake.condition == 'awake1')]['decay_isol'].median()

for animal in ['TS_C_AWAKE_CTRL_8235', 'TS_C_AWAKE_CTRL_8237', 'TS_C_AWAKE_CTRL_8238', 'TS_C_AWAKE_CTRL_F0','TS_C_AWAKE_CTRL_M3']:  
    sel_animal = (df_estimators_awake.animal == animal)
    awake1_median = df_estimators_awake[sel&sel_animal&(df_estimators_awake.condition == 'awake1')]['decay_isol'].median()
    df_estimators_awake.loc[sel&sel_animal, 'decay_isol_norm' ] /=  awake1_median
    
    
    ### STYLE

plt.rcParams['figure.figsize'] = [12, 8]
sns.set_context("paper", rc={"font.size":12,"axes.titlesize":12,"axes.labelsize":15})  
sns.set({"xtick.major.size": 12, "ytick.major.size": 12, 'xtick.labelsize': 'large', 'ytick.labelsize': 'large',
 'legend.fontsize': 'large'},    style="ticks")

### N_PEAKS_NPRM
    
# ZERO PEAKES EXCLUDED

sns.lineplot(x='time', y='n_peaks_norm', markers='o',lw=3,estimator=np.median,ci=95, color=color_keta,data=df_estimators_keta[(df_estimators_keta.n_peaks > 0.3)],label='Keta/Xyl')
sns.lineplot(x='time', y='n_peaks_norm', markers='o',lw=3,estimator=np.median,ci=95, color=color_mmf,data=df_estimators_mmf[(df_estimators_mmf.n_peaks > 0.3)],label='MMF')
sns.lineplot(x='time', y='n_peaks_norm', markers='o',lw=3,estimator=np.median,ci=95, color=color_iso,data=df_estimators_iso[(df_estimators_iso.n_peaks > 0.3)],label='Iso')
sns.lineplot(x='time', y='n_peaks_norm', markers='o',lw=3,estimator=np.median,ci=95, color=color_awake,data=df_estimators_awake[(df_estimators_awake.n_peaks > 0.3)],label='Awake')

#plt.plot([30,30],[0,2.0],'--k')
#plt.plot([90,90],[0,2.0],'--k')
plt.legend(fontsize=15)
plt.ylabel("number of transients per time unit",fontsize=15)
plt.title("relative change of number of transients",fontsize=22)
plt.xlabel("Time, [min]", fontsize=15)
plt.xlim([0.0, 360.0])

plt.savefig("n_peaks_norm.svg", format = 'svg', dpi=300)
plt.savefig('n_peaks_norm.png', format='png')

#fig, ax = plt.subplots() # or 
#plt.savefig('filename.eps', format='eps')

#plt.plot([0.0, 1.0],[0.0, 1.0])

### AMPLITUDE

sns.lineplot(x='time', y='height.median_norm', markers='o',lw=3,estimator=np.median,ci=95, color=color_mmf,data=df_estimators_mmf[(df_estimators_mmf.n_peaks > 0.3)],label='MMF')
sns.lineplot(x='time', y='height.median_norm', markers='o',lw=3,estimator=np.median,ci=95, color=color_keta,data=df_estimators_keta[(df_estimators_keta.n_peaks > 0.3)],label='Keta/Xyl')
sns.lineplot(x='time', y='height.median_norm', markers='o',lw=3,estimator=np.median,ci=95, color=color_iso,data=df_estimators_iso[(df_estimators_iso.n_peaks > 0.3)],label='Iso')

sns.lineplot(x='time', y='height.median_norm', markers='o',lw=3,estimator=np.median,ci=95, color=color_awake,data=df_estimators_awake[(df_estimators_awake.n_peaks > 0.3)],label='Awake')

#plt.plot([30,30],[0,2.0],'--k')
#plt.plot([90,90],[0,2.0],'--k')
plt.legend(fontsize=15)
plt.title("relative amplitude",fontsize=22)
plt.ylabel("amplitude",fontsize=15)
plt.xlabel("Time, [min]", fontsize=15)
#plt.ylim([0.5,1.5])
plt.xlim([0,360])

plt.savefig("height.median_norm.svg", format = 'svg', dpi=300)
#plt.savefig('height.median_norm.eps', format='eps')
plt.savefig('height.median_norm.png', format='png')

### DECAY_ISOL

#(df_estimators_keta.n_peaks > 0.3)&(df_estimators_keta.condition == 'keta')&(df_estimators_keta.decay_isol > 1/30.)

sns.lineplot(x='time', y='decay_isol_norm', markers='o',estimator=np.median,ci=95, color=color_mmf,data=df_estimators_mmf[(df_estimators_mmf.n_peaks > 0.3)&(df_estimators_mmf.decay_isol > 1/30.)],label='MMF')
sns.lineplot(x='time', y='decay_isol_norm', markers='o',estimator=np.median,ci=95, color=color_keta,data=df_estimators_keta[(df_estimators_keta.n_peaks > 0.3)&(df_estimators_keta.decay_isol > 1/30.)],label='Keta/Xyl')
sns.lineplot(x='time', y='decay_isol_norm', markers='o',estimator=np.median,ci=95, color=color_iso,data=df_estimators_iso[(df_estimators_iso.n_peaks > 0.3)&(df_estimators_iso.decay_isol > 1/30.)],label='Iso')

sns.lineplot(x='time', y='decay_isol_norm', markers='o',estimator=np.median,ci=95, color=color_awake,data=df_estimators_awake[(df_estimators_awake.n_peaks > 0.3)&(df_estimators_awake.decay_isol > 1/30.)],label='Awake')

#plt.plot([30,30],[0,2.0],'--k')
#plt.plot([90,90],[0,2.0],'--k')

plt.xlim([0,360])
plt.legend(fontsize=15)
plt.title("relative decay constant",fontsize=22)
plt.ylabel("relative decay constant",fontsize=15)
plt.xlabel("time (min)", fontsize=15)
#plt.ylim([0.5,1.5])

plt.savefig("decay_isol_norm.svg", format = 'svg', dpi=300)
#plt.savefig('height.median_norm.eps', format='eps')
plt.savefig('decay_isol_norm.png', format='png')


for animal in ['TS_C_MMF_8235', 'TS_C_MMF_8237', 'TS_C_MMF_8238', 'TS_C_MMF_F0','TS_C_MMF_M0']:
    sns.lineplot(x='time', y='n_peaks', markers='o', estimator=np.median,ci=95,data=df_estimators_mmf[(df_estimators_mmf.n_peaks > 0.3)&(df_estimators_mmf.animal == animal)],label=animal)
plt.legend(fontsize=15)


for animal in ['TS_C_MMF_8235', 'TS_C_MMF_8237', 'TS_C_MMF_8238', 'TS_C_MMF_F0','TS_C_MMF_M0']:
    sns.lineplot(x='time', y='n_peaks_norm', markers='o', estimator=np.median,ci=95,data=df_estimators_mmf[(df_estimators_mmf.n_peaks > 0.3)&(df_estimators_mmf.animal == animal)],label=animal)
plt.legend(fontsize=15)


for animal in ['TS_C_MMF_8235', 'TS_C_MMF_8237', 'TS_C_MMF_8238', 'TS_C_MMF_F0','TS_C_MMF_M0']:
    sns.lineplot(x='time', y='height.median', markers='o', estimator=np.median,ci=95,data=df_estimators_mmf[df_estimators_mmf.animal == animal],label=animal)
plt.legend(fontsize=15)


for animal in ['TS_C_MMF_8235', 'TS_C_MMF_8237', 'TS_C_MMF_8238', 'TS_C_MMF_F0','TS_C_MMF_M0']:
    sns.lineplot(x='time', y='decay_isol', markers='o', estimator=np.median,ci=95,data=df_estimators_mmf[df_estimators_mmf.animal == animal],label=animal)
plt.legend(fontsize=15)


for animal in ['TS_C_MMF_8235', 'TS_C_MMF_8237', 'TS_C_MMF_8238', 'TS_C_MMF_F0','TS_C_MMF_M0']:
    sns.lineplot(x='time', y='baseline.oasis', markers='o', estimator=np.median,ci=95,data=df_estimators_mmf[df_estimators_mmf.animal == animal],label=animal)
plt.legend(fontsize=15)


for animal in ['TS_C_ISO_8235', 'TS_C_ISO_8237', 'TS_C_ISO_8238', 'TS_C_ISO_F0','TS_C_ISO_M0']:
    sns.lineplot(x='time', y='n_peaks', markers='o', estimator=np.median,data=df_estimators_iso[df_estimators_iso.animal == animal],label=animal)
plt.legend(fontsize=15)



for animal in ['TS_C_ISO_8235', 'TS_C_ISO_8237', 'TS_C_ISO_8238', 'TS_C_ISO_F0','TS_C_ISO_M0']:
    sns.lineplot(x='time', y='baseline.oasis', markers='o', estimator=np.median,data=df_estimators_iso[df_estimators_iso.animal == animal],label=animal)
plt.legend(fontsize=15)


for animal in ['TS_C_KETA_8235', 'TS_C_KETA_8237', 'TS_C_KETA_8238', 'TS_C_KETA_F0','TS_C_KETA_M3']:
    sns.lineplot(x='time', y='n_peaks', markers='o', estimator=np.median,data=df_estimators_keta[df_estimators_keta.animal == animal],label=animal)
plt.legend(fontsize=15)


for animal in ['TS_C_KETA_8235', 'TS_C_KETA_8237', 'TS_C_KETA_8238', 'TS_C_KETA_F0','TS_C_KETA_M3']:
    sns.lineplot(x='time', y='height.median', markers='o', estimator=np.median,data=df_estimators_keta[df_estimators_keta.animal == animal],label=animal)
plt.legend(fontsize=15)

for animal in ['TS_C_KETA_8235', 'TS_C_KETA_8237', 'TS_C_KETA_8238', 'TS_C_KETA_F0','TS_C_KETA_M3']:
    sns.lineplot(x='time', y='height.median_norm', markers='o', estimator=np.median,data=df_estimators_keta[df_estimators_keta.animal == animal],label=animal)
plt.legend(fontsize=15)

for animal in ['TS_C_KETA_8235', 'TS_C_KETA_8237', 'TS_C_KETA_8238', 'TS_C_KETA_F0','TS_C_KETA_M3']:
    sns.lineplot(x='time', y='baseline.oasis', markers='o', estimator=np.median,data=df_estimators_keta[df_estimators_keta.animal == animal],label=animal)
plt.legend(fontsize=15)

for animal in ['TS_C_KETA_8235', 'TS_C_KETA_8237', 'TS_C_KETA_8238', 'TS_C_KETA_M3']:
    sns.lineplot(x='time', y='decay_isol', markers='o', estimator=np.median,data=df_estimators_keta[(df_estimators_keta.animal == animal)&(df_estimators_keta.decay_isol < 25)],label=animal)
plt.legend(fontsize=15)

for animal in ['TS_C_KETA_8235', 'TS_C_KETA_8237', 'TS_C_KETA_8238', 'TS_C_KETA_M3']:
    sns.lineplot(x='time', y='decay_isol', markers='o', estimator=np.median,data=df_estimators_keta[(df_estimators_keta.animal == animal)&(df_estimators_keta.decay_isol < 25)],label=animal)
plt.legend(fontsize=15)


from scipy.stats import gmean

for animal in ['TS_C_AWAKE_CTRL_8235', 'TS_C_AWAKE_CTRL_8237', 'TS_C_AWAKE_CTRL_8238','TS_C_AWAKE_CTRL_F0']:
    sns.lineplot(x='time', y='n_peaks_norm', markers='o', estimator=np.median,data=df_estimators_awake[df_estimators_awake.animal == animal], label=animal)
plt.legend(fontsize=15)
#plt.yscale('log')
#plt.ylim([0.01,100])

for animal in ['TS_C_AWAKE_CTRL_8235', 'TS_C_AWAKE_CTRL_8237', 'TS_C_AWAKE_CTRL_8238','TS_C_AWAKE_CTRL_F0']:
    sns.lineplot(x='time', y='n_peaks', markers='o', estimator=np.median,data=df_estimators_awake[df_estimators_awake.animal == animal], label=animal)
plt.legend(fontsize=15)


for animal in ['TS_C_AWAKE_CTRL_8235', 'TS_C_AWAKE_CTRL_8237', 'TS_C_AWAKE_CTRL_8238','TS_C_AWAKE_CTRL_F0']:
    sns.lineplot(x='time', y='trace.std', markers='o', estimator=np.median,data=df_estimators_awake[(df_estimators_awake.animal == animal)&(df_estimators_awake.n_peaks > 0.3)], label=animal)
plt.legend(fontsize=15)


for animal in ['TS_C_AWAKE_CTRL_8235', 'TS_C_AWAKE_CTRL_8237', 'TS_C_AWAKE_CTRL_8238','TS_C_AWAKE_CTRL_F0']:
    sns.lineplot(x='time', y='traces.median', markers='o', estimator=np.median,data=df_estimators_awake[(df_estimators_awake.animal == animal)&(df_estimators_awake.n_peaks > 0.3)], label=animal)
plt.legend(fontsize=15)


for animal in ['TS_C_AWAKE_CTRL_8235', 'TS_C_AWAKE_CTRL_8237', 'TS_C_AWAKE_CTRL_8238', 'TS_C_AWAKE_CTRL_F0']:
    sns.lineplot(x='time', y='height.median', markers='o', estimator=np.median,data=df_estimators_awake[df_estimators_awake.animal == animal], label=animal)
plt.legend(fontsize=15)

for animal in ['TS_C_AWAKE_CTRL_8235', 'TS_C_AWAKE_CTRL_8237', 'TS_C_AWAKE_CTRL_8238', 'TS_C_AWAKE_CTRL_F0']:
    sns.lineplot(x='time', y='baseline.oasis', markers='o', estimator=np.median,data=df_estimators_awake[df_estimators_awake.animal == animal], label=animal)
plt.legend(fontsize=15)

for animal in ['TS_C_AWAKE_CTRL_8235', 'TS_C_AWAKE_CTRL_8237', 'TS_C_AWAKE_CTRL_8238', 'TS_C_AWAKE_CTRL_F0']:
    sns.lineplot(x='time', y='decay_isol', markers='o', estimator=np.median,data=df_estimators_awake[df_estimators_awake.animal == animal], label=animal)
plt.legend(fontsize=15)

###################################################################
############ HISTOGRAMS
###################################################################

sns.displot(data=df_estimators_awake[(df_estimators_awake.animal == 'TS_C_AWAKE_CTRL_8238')&(df_estimators_awake.condition == 'awake1')&(df_estimators_awake.n_peaks > 0.3)], x='n_peaks', bins=30)
#plt.xlim([0, 125])

estimators_awake.animal == 'TS_C_AWAKE_CTRL_8238')&(df_estimators_awake.condition == 'awake1')&(df_estimators_awake.n_peaks > 0.3)], x='trace.std', bins=30, log_scale=True)

sns.displot(data=df_estimators_awake[(df_estimators_awake.animal == 'TS_C_AWAKE_CTRL_8238')&(df_estimators_awake.condition == 'awake2')&(df_estimators_awake.n_peaks > 0.3)], x='trace.std', bins=30, log_scale=True)
sns.displot(data=df_estimators_awake[(df_estimators_awake.animal == 'TS_C_AWAKE_CTRL_8238')&(df_estimators_awake.condition == 'awake3')&(df_estimators_awake.n_peaks > 0.3)], x='trace.std', bins=30, log_scale=True)
sns.displot(data=df_estimators_awake[(df_estimators_awake.animal == 'TS_C_AWAKE_CTRL_8238')&(df_estimators_awake.condition == 'awake4')&(df_estimators_awake.n_peaks > 0.3)], x='trace.std', bins=30, log_scale=True)

sns.displot(data=df_estimators_awake[(df_estimators_awake.animal == 'TS_C_AWAKE_CTRL_8238')&(df_estimators_awake.condition == 'awake5')&(df_estimators_awake.n_peaks > 0.3)], x='trace.std', bins=30, log_scale=True)
sns.displot(data=df_estimators_awake[(df_estimators_awake.animal == 'TS_C_AWAKE_CTRL_8238')&(df_estimators_awake.condition == 'awake6')&(df_estimators_awake.n_peaks > 0.3)], x='trace.std', bins=30, log_scale=True)

#plt.xlim([0, 125])

#(df_estimators_awake.animal == 'TS_C_AWAKE_CTRL_F0')&
#(df_estimators_awake.condition == 'awake1')&
sns.displot(data=df_estimators_awake[(df_estimators_awake.n_peaks > 0.1)], x='n_peaks', hue='condition', bins=30, log_scale=False)

print(df_estimators_awake[(df_estimators_awake.condition == 'awake1')&(df_estimators_awake.n_peaks > 0.3)]['n_peaks'].median())
print(df_estimators_awake[(df_estimators_awake.condition == 'awake2')&(df_estimators_awake.n_peaks > 0.3)]['n_peaks'].median())
print(df_estimators_awake[(df_estimators_awake.condition == 'awake3')&(df_estimators_awake.n_peaks > 0.3)]['n_peaks'].median())
print(df_estimators_awake[(df_estimators_awake.condition == 'awake4')&(df_estimators_awake.n_peaks > 0.3)]['n_peaks'].median())
print(df_estimators_awake[(df_estimators_awake.condition == 'awake5')&(df_estimators_awake.n_peaks > 0.3)]['n_peaks'].median())
print(df_estimators_awake[(df_estimators_awake.condition == 'awake6')&(df_estimators_awake.n_peaks > 0.3)]['n_peaks'].median())


################################################################
### PCA all-at-one-plot   (Including PCA does not make any sence!)
################################################################

### INCLUSION OF RECOVERY RECORDING (60 min) DISTINGUISH AWAKE AND ANESTHESIA CONDITION, SPLITS ISO, KETA AND MMF
 

### http://alexhwilliams.info/itsneuronalblog/2015/09/11/clustering1/

from sklearn.decomposition import PCA

df_estimators_iso['CONDITION'] = 'ISO'
df_estimators_mmf['CONDITION'] = 'MMF'
df_estimators_keta['CONDITION'] = 'KETA'

df_estimators_awake['CONDITION'] = 'CTRL'

df_estimators_awake_copy = df_estimators_awake.copy()

df_estimators_awake_copy.loc[df_estimators_awake_copy.time == 60,'time'] = 30

df_estimators_PCA = pd.concat([df_estimators_iso,df_estimators_keta,df_estimators_mmf,df_estimators_awake_copy], copy=True)


#df_estimators_PCA = pd.concat([df_estimators_iso,df_estimators_keta,df_estimators_mmf], copy=True)

df_estimators_PCA.loc[df_estimators_PCA.time == 0,'CONDITION'] = 'AWAKE'


select_times = (df_estimators_PCA.time == 0)|(df_estimators_PCA.time == 30) #|(df_estimators_PCA.time == 60)


df_estimators_PCA = df_estimators_PCA[select_times&(df_estimators_PCA.n_peaks > 0.3)&(df_estimators_PCA.decay_isol > 1/30.)&(df_estimators_PCA.decay_isol < 10)].copy()


#df_estimators_PCA.loc[df_estimators_PCA.time == 0,'CONDITION'] = 'AWAKE'

df_estimators_PCA = df_estimators_PCA[['n_peaks_norm','height.median_norm','decay_isol_norm','animal','CONDITION','time']].copy()

#df_estimators_PCA.fillna(0, inplace=True) 

print(df_estimators_PCA.isnull().any()) # Check for NaN 


pca = PCA(n_components=3, whiten=True)

X = df_estimators_PCA[['n_peaks_norm','height.median_norm','decay_isol_norm']].to_numpy().copy()


pca.fit(X)
Xpca = pca.transform(X)




df_estimators_PCA['PCA1'] = Xpca[:,0]
df_estimators_PCA['PCA2'] = Xpca[:,1]
df_estimators_PCA['PCA3'] = Xpca[:,2]




print("PCA1 %.2f %%" % (pca.explained_variance_ratio_[0]*100))
print("PCA2 %.2f %%" % (pca.explained_variance_ratio_[1]*100))
print("PCA3 %.2f %%" % (pca.explained_variance_ratio_[2]*100))


plt.rcParams['figure.figsize'] = [5, 5]

plt.figure()

df_est_PCA_median = df_estimators_PCA[(df_estimators_PCA.time == 0)|(df_estimators_PCA.time == 30)].groupby(['animal','CONDITION'])[['PCA1','PCA2']].median()

sns.scatterplot(x='PCA1', y='PCA2',  data=df_est_PCA_median, style="animal", hue="CONDITION", palette=(color_awake,color_ctrl,color_iso,color_keta,color_mmf), s=150, alpha = 1, legend=False)


plt.savefig("PCA_2states.svg", format = 'svg', dpi=300)
plt.savefig("PCA_2states.png", format='png')



################################################################
### PCA three - plots
################################################################

from sklearn.decomposition import PCA

df_estimators_iso['CONDITION'] = 'ISO'
df_estimators_mmf['CONDITION'] = 'MMF'
df_estimators_keta['CONDITION'] = 'KETA'
df_estimators_awake['CONDITION'] = 'CTRL'

df_estimators_awake_copy = df_estimators_awake.copy()

df_estimators_awake_copy.loc[df_estimators_awake_copy.time == 60,'time'] = 30

df_estimators_PCA = pd.concat([df_estimators_iso,df_estimators_keta,df_estimators_mmf,df_estimators_awake_copy], copy=True)


df_estimators_PCA = df_estimators_PCA[(df_estimators_PCA.n_peaks > 0.3)&(df_estimators_PCA.decay_isol > 1/30.)&(df_estimators_PCA.decay_isol < 10)].copy()

#df_estimators_PCA.loc[df_estimators_PCA.time == 0,'CONDITION'] = 'AWAKE'

df_estimators_PCA = df_estimators_PCA[['n_peaks_norm','height.median_norm','decay_isol_norm','animal','CONDITION','time']].copy()

#df_estimators_PCA.fillna(0, inplace=True) 

print(df_estimators_PCA.isnull().any()) # Check for NaN 



pca = PCA(n_components=3, whiten=True)

X = df_estimators_PCA[['n_peaks_norm','height.median_norm','decay_isol_norm']].to_numpy().copy()

pca.fit(X)
Xpca = pca.transform(X)


df_estimators_PCA['PCA1'] = Xpca[:,0]
df_estimators_PCA['PCA2'] = Xpca[:,1]
df_estimators_PCA['PCA3'] = Xpca[:,2]


print("PCA1 %.2f %%" % (pca.explained_variance_ratio_[0]*100))
print("PCA2 %.2f %%" % (pca.explained_variance_ratio_[1]*100))
print("PCA3 %.2f %%" % (pca.explained_variance_ratio_[2]*100))


#Plot without different markers, sns.scatterplot(x='PCA1', y='PCA2',  data=df_est_PCA_median, hue="CONDITION", palette=(color_awake,color_iso,color_keta,color_mmf), s=250, alpha = 0.8)


import scipy.interpolate as si

def bspline(cv, n=100, degree=2):
    """ Calculate n samples on a bspline

        cv :      Array ov control vertices
        n  :      Number of samples to return
        degree:   Curve degree
    """
    cv = np.asarray(cv)
    count = cv.shape[0]

    # Prevent degree from exceeding count-1, otherwise splev will crash
    degree = np.clip(degree,1,count-1)

    # Calculate knot vector
    kv = np.array([0]*degree + list(range(count-degree+1)) + [count-degree]*degree,dtype='int')

    # Calculate query range
    u = np.linspace(0,(count-degree),n)

    # Calculate result
    return np.array(si.splev(u, (kv,cv.T,degree))).T


######### KETA ##########

plt.rcParams['figure.figsize'] = [10, 10]

plt.figure()


############# BSPLINES ##################

for animal in np.unique(df_estimators_keta.animal):
    df_est_PCA_median_traject = df_estimators_PCA[(df_estimators_PCA.CONDITION == 'KETA')&(df_estimators_PCA.animal == animal)].groupby(['time'])[['PCA1','PCA2','CONDITION']].median()
    x=df_est_PCA_median_traject.PCA1.to_numpy()[1:]
    y=df_est_PCA_median_traject.PCA2.to_numpy()[1:]

    cv = np.concatenate((x,y)).reshape(2,8).T.copy()

    #plt.plot(cv[:,0],cv[:,1], 'o-', label='Control Points')
    p = bspline(cv)
    xs,ys = p.T
    plt.plot(xs,ys,'k-',lw=3,color=color_keta)


df_est_PCA_median = df_estimators_PCA[(df_estimators_PCA.CONDITION == 'KETA')&(df_estimators_PCA.time == 0)].groupby(['animal','CONDITION'])[['PCA1','PCA2']].median()

sns.scatterplot(x='PCA1', y='PCA2',  data=df_est_PCA_median, style="animal", hue="CONDITION", palette=[color_awake], s=250, alpha = 1)


df_est_PCA_median = df_estimators_PCA[(df_estimators_PCA.CONDITION == 'KETA')&(df_estimators_PCA.time == 30)].groupby(['animal','CONDITION'])[['PCA1','PCA2']].median()

sns.scatterplot(x='PCA1', y='PCA2',  data=df_est_PCA_median, style="animal", hue="CONDITION", palette=[color_keta], s=250, alpha = 1)

df_est_PCA_median = df_estimators_PCA[(df_estimators_PCA.CONDITION == 'KETA')&(df_estimators_PCA.time != 0)&(df_estimators_PCA.time != 30)].groupby(['animal','time'])[['PCA1','PCA2','CONDITION']].median()

sns.scatterplot(x='PCA1', y='PCA2',  data=df_est_PCA_median, style="animal", hue="time",color=[color_keta], s=100, alpha = 0.5)  #,palette=(color_awake,color_iso,color_keta,color_mmf)

plt.ylim([-2,1.3])

plt.xlim([-0.38,0.45])


plt.legend([],[], frameon=False)

#plt.title("Keta/Xyl")

plt.savefig("PCA_recovery_keta.svg", format = 'svg', dpi=300)
plt.savefig("PCA_recovery_keta.png", format='png')

############# MMF ############

plt.rcParams['figure.figsize'] = [10, 10]

plt.figure()

############# BSPLINES ##################

for animal in np.unique(df_estimators_mmf.animal):
    df_est_PCA_median_traject = df_estimators_PCA[(df_estimators_PCA.CONDITION == 'MMF')&(df_estimators_PCA.animal == animal)].groupby(['time'])[['PCA1','PCA2','CONDITION']].median()
    x=df_est_PCA_median_traject.PCA1.to_numpy()[1:]
    y=df_est_PCA_median_traject.PCA2.to_numpy()[1:]

    cv = np.concatenate((x,y)).reshape(2,8).T.copy()

    #plt.plot(cv[:,0],cv[:,1], 'o-', label='Control Points')
    p = bspline(cv)
    xs,ys = p.T
    plt.plot(xs,ys,'k-',lw=3,color=color_mmf)


df_est_PCA_median = df_estimators_PCA[(df_estimators_PCA.CONDITION == 'MMF')&(df_estimators_PCA.time == 0)].groupby(['animal','CONDITION'])[['PCA1','PCA2']].median()

sns.scatterplot(x='PCA1', y='PCA2',  data=df_est_PCA_median, style="animal", hue="CONDITION", palette=[color_awake], s=250, alpha = 1)

df_est_PCA_median = df_estimators_PCA[(df_estimators_PCA.CONDITION == 'MMF')&(df_estimators_PCA.time == 30)].groupby(['animal','CONDITION'])[['PCA1','PCA2']].median()

sns.scatterplot(x='PCA1', y='PCA2',  data=df_est_PCA_median, style="animal", hue="CONDITION", palette=[color_mmf], s=250, alpha = 1)

df_est_PCA_median = df_estimators_PCA[(df_estimators_PCA.CONDITION == 'MMF')&(df_estimators_PCA.time != 0)&(df_estimators_PCA.time != 30)].groupby(['animal','time'])[['PCA1','PCA2','CONDITION']].median()

sns.scatterplot(x='PCA1', y='PCA2',  data=df_est_PCA_median, style="animal", hue="time",color=[color_awake], s=100, alpha = 0.5)  #,palette=(color_awake,color_iso,color_keta,color_mmf)

plt.ylim([-2,1.3])

plt.xlim([-0.38,0.45])

plt.legend([],[], frameon=False)


plt.savefig("PCA_recovery_mmf.svg", format = 'svg', dpi=300)
plt.savefig("PCA_recovery_mmf.png", format='png')

######### ISO ##########

plt.figure()

############# BSPLINES ##################

for animal in np.unique(df_estimators_iso.animal):
    df_est_PCA_median_traject = df_estimators_PCA[(df_estimators_PCA.CONDITION == 'ISO')&(df_estimators_PCA.animal == animal)].groupby(['time'])[['PCA1','PCA2','CONDITION']].median()
    x=df_est_PCA_median_traject.PCA1.to_numpy()[1:]
    y=df_est_PCA_median_traject.PCA2.to_numpy()[1:]

    cv = np.concatenate((x,y)).reshape(2,8).T.copy()

    #plt.plot(cv[:,0],cv[:,1], 'o-', label='Control Points')
    p = bspline(cv)
    xs,ys = p.T
    plt.plot(xs,ys,'k-',lw=3,color=color_iso)

df_est_PCA_median = df_estimators_PCA[(df_estimators_PCA.CONDITION == 'ISO')&(df_estimators_PCA.time == 0)].groupby(['animal','CONDITION'])[['PCA1','PCA2']].median()

sns.scatterplot(x='PCA1', y='PCA2',  data=df_est_PCA_median, style="animal", hue="CONDITION", palette=[color_awake], s=250, alpha = 1)

df_est_PCA_median = df_estimators_PCA[(df_estimators_PCA.CONDITION == 'ISO')&(df_estimators_PCA.time == 30)].groupby(['animal','CONDITION'])[['PCA1','PCA2']].median()

sns.scatterplot(x='PCA1', y='PCA2',  data=df_est_PCA_median, style="animal", hue="CONDITION", palette=[color_iso], s=250, alpha = 1)

df_est_PCA_median = df_estimators_PCA[(df_estimators_PCA.CONDITION == 'ISO')&(df_estimators_PCA.time != 0)&(df_estimators_PCA.time != 30)].groupby(['animal','time'])[['PCA1','PCA2','CONDITION']].median()

sns.scatterplot(x='PCA1', y='PCA2',  data=df_est_PCA_median, style="animal", hue="time",color=[color_iso], s=100, alpha = 0.5)  #,palette=(color_awake,color_iso,color_keta,color_mmf)

plt.ylim([-2,1.3])

plt.xlim([-0.38,0.45])

plt.legend([],[], frameon=False)


plt.savefig("PCA_recovery_iso.svg", format = 'svg', dpi=300)
plt.savefig("PCA_recovery_iso.png", format='png')


######### AWAKE (CTLR) ##########

plt.rcParams['figure.figsize'] = [10, 10]

plt.figure()

############# BSPLINES ##################

for animal in np.unique(df_estimators_awake_copy.animal):
    df_est_PCA_median_traject = df_estimators_PCA[(df_estimators_PCA.CONDITION == 'CTRL')&(df_estimators_PCA.animal == animal)].groupby(['time'])[['PCA1','PCA2','CONDITION']].median()
    x=df_est_PCA_median_traject.PCA1.to_numpy()[1:]
    y=df_est_PCA_median_traject.PCA2.to_numpy()[1:]
    x = x[0:6]
    y = y[0:6]
    cv = np.concatenate((x,y)).reshape(2,6).T.copy()

    #plt.plot(cv[:,0],cv[:,1], 'o-', label='Control Points')
    p = bspline(cv)
    xs,ys = p.T
    plt.plot(xs,ys,'k-',lw=3,color=color_ctrl)


df_est_PCA_median = df_estimators_PCA[(df_estimators_PCA.CONDITION == 'CTRL')&(df_estimators_PCA.time == 0)].groupby(['animal','CONDITION'])[['PCA1','PCA2']].median()

sns.scatterplot(x='PCA1', y='PCA2',  data=df_est_PCA_median, style="animal", hue="CONDITION", palette=[color_awake], s=250, alpha = 1)

df_est_PCA_median = df_estimators_PCA[(df_estimators_PCA.CONDITION == 'CTRL')&(df_estimators_PCA.time == 30)].groupby(['animal','CONDITION'])[['PCA1','PCA2']].median()

sns.scatterplot(x='PCA1', y='PCA2',  data=df_est_PCA_median, style="animal", hue="CONDITION", palette=[color_ctrl], s=250, alpha = 1)

df_est_PCA_median = df_estimators_PCA[(df_estimators_PCA.CONDITION == 'CTRL')&(df_estimators_PCA.time != 0)&(df_estimators_PCA.time != 30)].groupby(['animal','time'])[['PCA1','PCA2','CONDITION']].median()

sns.scatterplot(x='PCA1', y='PCA2',  data=df_est_PCA_median, style="animal", hue="time",color=[color_keta], s=100, alpha = 0.5)  #,palette=(color_awake,color_iso,color_keta,color_mmf)

plt.ylim([-2,1.3])

plt.xlim([-0.38,0.45])

plt.legend([],[], frameon=False)

plt.savefig("PCA_recovery_ctrl.svg", format = 'svg', dpi=300)
plt.savefig("PCA_recovery_ctrl.png", format='png')


###############  ANIMATION #####################################
################################################################



################################################################
### PCA
################################################################

#PCA

df_estimators_iso['CONDITION'] = 'ISO'
df_estimators_mmf['CONDITION'] = 'MMF'
df_estimators_keta['CONDITION'] = 'KETA'
#df_estimators_awake['CONDITION'] = 'CTRL'

df_estimators_PCA = pd.concat([df_estimators_iso,df_estimators_keta,df_estimators_mmf], copy=True)

#df_estimators_PCA[df_estimators_PCA.time == 0]['CONDITION','time'] = 'AWAKE'

df_estimators_PCA.loc[df_estimators_PCA.time == 0,'CONDITION'] = 'AWAKE'

#df.loc[df['First Season'] > 1990, 'First Season'] = 1

sel_tain_test = (df_estimators_PCA.time == 0)|(df_estimators_PCA.time == 60)

df_estimators_PCA_train = df_estimators_PCA[sel_tain_test]

df_estimators_PCA_test = df_estimators_PCA[~sel_tain_test]

df_estimators_PCA_train = df_estimators_PCA_train[['n_peaks_norm','height.median_norm','decay_isol_norm','animal','CONDITION','time']].copy()

df_estimators_PCA_test = df_estimators_PCA_test[['n_peaks_norm','height.median_norm','decay_isol_norm','animal','CONDITION','time']].copy()

#df_estimators_PCA = 
# df_estimators_mmf #pd.concat([df_estimators_mmf,df_estimators_iso,df_estimators_keta])
df_estimators_PCA_train.fillna(0, inplace=True) 
df_estimators_PCA_test.fillna(0, inplace=True) 

plt.rcParams['figure.figsize'] = [20, 20]

from sklearn.decomposition import PCA

pca = PCA(n_components=2,whiten=True)

X = df_estimators_PCA_train[['n_peaks_norm','height.median_norm','decay_isol_norm']].to_numpy().copy()
X_test = df_estimators_PCA_test[['n_peaks_norm','height.median_norm','decay_isol_norm']].to_numpy().copy()

pca.fit(X)
Xpca = pca.transform(X)

Xpca_test = pca.transform(X_test)

df_estimators_PCA_train['PCA1'] = Xpca[:,0]
df_estimators_PCA_train['PCA2'] = Xpca[:,1]
#df_estimators_PCA_train['PCA3'] = Xpca[:,2]

df_estimators_PCA_test['PCA1'] = Xpca_test[:,0]
df_estimators_PCA_test['PCA2'] = Xpca_test[:,1]
#df_estimators_PCA_test['PCA3'] = Xpca_test[:,2]

print("PCA1 %.2f %%" % (pca.explained_variance_ratio_[0]*100))
print("PCA2 %.2f %%" % (pca.explained_variance_ratio_[1]*100))
#selector = (df_estimators_PCA['animal'] == 'TS_C_MMF_M0')
#selector = (df_estimators_PCA['CONDITION'] == 'ISO')
#selector = 1

#df_est_PCA_median = df_estimators_PCA[df_estimators_PCA.time == 0].groupby(['animal','time','CONDITION'])[['PCA1','PCA2']].median()
#sns.scatterplot(x='PCA1', y='PCA2', s = 15, data=df_est_PCA_median,size='time',hue="CONDITION",sizes=[10,500,360,300,240,120,90,60,30], alpha=.5)

#selector = (df_estimators_PCA['animal'] == 'TS_C_MMF_M0')

df_est_PCA_median = df_estimators_PCA_train.groupby(['animal','CONDITION'])[['PCA1','PCA2']].mean()

df_est_PCA_median_test = df_estimators_PCA_test.groupby(['animal','time'])[['PCA1','PCA2','CONDITION']].mean()


#Plot without different markers, sns.scatterplot(x='PCA1', y='PCA2',  data=df_est_PCA_median, hue="CONDITION", palette=(color_awake,color_iso,color_keta,color_mmf), s=250, alpha = 0.8)

plt.figure()
df_estimators_iso['CONDITION'] = 'ISO'
df_estimators_mmf['CONDITION'] = 'MMF'
df_estimators_keta['CONDITION'] = 'KETA'
#df_estimators_awake['CONDITION'] = 'CTRL'

df_estimators_PCA = pd.concat([df_estimators_iso,df_estimators_keta,df_estimators_mmf], copy=True)

#df_estimators_PCA[df_estimators_PCA.time == 0]['CONDITION','time'] = 'AWAKE'

df_estimators_PCA.loc[df_estimators_PCA.time == 0,'CONDITION'] = 'AWAKE'

#df.loc[df['First Season'] > 1990, 'First Season'] = 1

sel_tain_test = (df_estimators_PCA.time == 0)|(df_estimators_PCA.time == 60)

df_estimators_PCA_train = df_estimators_PCA[sel_tain_test]

df_estimators_PCA_test = df_estimators_PCA[~sel_tain_test]

df_estimators_PCA_train = df_estimators_PCA_train[['n_peaks_norm','height.median_norm','decay_isol_norm','animal','CONDITION','time']].copy()

df_estimators_PCA_test = df_estimators_PCA_test[['n_peaks_norm','height.median_norm','decay_isol_norm','animal','CONDITION','time']].copy()

#df_estimators_PCA = 
# df_estimators_mmf #pd.concat([df_estimators_mmf,df_estimators_iso,df_estimators_keta])
df_estimators_PCA_train.fillna(0, inplace=True) 
df_estimators_PCA_test.fillna(0, inplace=True) 

sns.scatterplot(x='PCA1', y='PCA2',  data=df_est_PCA_median, style="animal", hue="CONDITION",palette=(color_awake,color_iso,color_keta,color_mmf), s=250, alpha = 1)

#sns.scatterplot(x='PCA1', y='PCA2',  data=df_est_PCA_median_test, style="animal", hue="time", s=250, alpha = 0.6)  #,palette=(color_awake,color_iso,color_keta,color_mmf)

#lgnd = plt.legend() #(loc="lower left", scatterpoints=1, fontsize=10)
#lgnd.legendHandles[0]._sizes = [300,300]
#lgnd.legendHandles[1]._sizes = [300,300]

plt.xlim([-0.32-0.02,-0.27+0.01])
plt.ylim([-0.7-0.1,0.0])
plt.show()
   





### Plot trajectories

df_est_PCA_median = df_estimators_PCA_train.groupby(['animal','condition'])[['PCA1','PCA2']].median()

df_est_PCA_median_test = df_estimators_PCA_test.groupby(['animal','time'])[['PCA1','PCA2','condition']].median()

plt.figure()

sns.scatterplot(x='PCA1', y='PCA2',  data=df_est_PCA_median[df_est_PCA_median.condition=='iso'], style="animal", hue="condition",palette=(color_awake,'k',color_iso,color_keta,color_mmf), s=250, alpha = 1)



######################################################################################



################################################################################################
######### t-SNE
################################################################################################

from sklearn.manifold import TSNE

df_estimators_iso['CONDITION'] = 'ISO'
df_estimators_mmf['CONDITION'] = 'MMF'
df_estimators_keta['CONDITION'] = 'KETA'
#df_estimators_awake['CONDITION'] = 'CTRL'

df_estimators_tSNE = pd.concat([df_estimators_iso,df_estimators_keta,df_estimators_mmf], copy=True)

df_estimators_tSNE = df_estimators_tSNE[(df_estimators_tSNE.n_peaks > 0.3)&(df_estimators_tSNE.decay_isol > 1/30.)&(df_estimators_tSNE.decay_isol < 10)].copy()

sel_tain_test = (df_estimators_tSNE.time == 0)|(df_estimators_tSNE.time == 30)

df_estimators_tSNE = df_estimators_tSNE[sel_tain_test].copy()

df_estimators_tSNE.loc[df_estimators_tSNE.time == 0,'CONDITION'] = 'AWAKE'

#df_estimators_PCA.loc[df_estimators_PCA.time == 0,'CONDITION'] = 'AWAKE'

df_estimators_tSNE= df_estimators_tSNE[['n_peaks_norm','height.median_norm','decay_isol_norm','animal','CONDITION','condition','time']].copy()

tsne = TSNE(random_state=0)

df_estimators_tSNE.fillna(0, inplace=True) 

X = df_estimators_tSNE[['n_peaks_norm','height.median_norm','decay_isol_norm']].to_numpy().copy()

Y = tsne.fit_transform(X)

df_estimators_tSNE['tSNE1'] = Y[:,0]

df_estimators_tSNE['tSNE2'] = Y[:,1]


plt.scatter(Y[:,0],Y[:,1])


plt.figure()

sns.scatterplot(x='tSNE1', y='tSNE2',  data=df_estimators_tSNE, style="animal", hue="CONDITION", palette=(color_awake,color_ctrl,color_iso,color_keta,color_mmf), s=150, alpha = 1, legend=False)


sns.scatterplot(x='tSNE1', y='tSNE2',  data=df_estimators_tSNE[df_estimators_tSNE.animal == "TS_C_KETA_8235"], style="animal", hue="CONDITION", palette=(color_awake,color_keta), s=50, alpha = 0.6, legend=True)


sns.scatterplot(x='tSNE1', y='tSNE2',  data=df_estimators_tSNE[df_estimators_tSNE.animal == "TS_C_KETA_8238"], style="animal", hue="CONDITION", palette=(color_awake,color_keta), s=50, alpha = 0.6, legend=True)


sns.scatterplot(x='tSNE1', y='tSNE2',  data=df_estimators_tSNE[df_estimators_tSNE.animal == "TS_C_KETA_8237"], style="animal", hue="CONDITION", palette=(color_awake,color_keta), s=50, alpha = 0.6, legend=True)


sns.scatterplot(x='tSNE1', y='tSNE2',  data=df_estimators_tSNE[df_estimators_tSNE.animal == "TS_C_MMF_8235"], style="animal", hue="CONDITION", palette=(color_awake,color_mmf), s=50, alpha = 0.6, legend=True)


sns.scatterplot(x='tSNE1', y='tSNE2',  data=df_estimators_tSNE[df_estimators_tSNE.animal == "TS_C_MMF_8238"], style="animal", hue="CONDITION", palette=(color_awake,color_mmf), s=50, alpha = 0.6, legend=True)


sns.scatterplot(x='tSNE1', y='tSNE2',  data=df_estimators_tSNE[df_estimators_tSNE.animal == "TS_C_MMF_8237"], style="animal", hue="CONDITION", palette=(color_awake,color_mmf), s=50, alpha = 0.6, legend=True)

sns.scatterplot(x='tSNE1', y='tSNE2',  data=df_estimators_tSNE[df_estimators_tSNE.CONDITION == "KETA"], style="animal", hue="CONDITION",  s=150, alpha = 1, legend=False)


#df_est_tSNE_median = df_estimators_PCA[(df_estimators_PCA.time == 0)|(df_estimators_PCA.time == 30)].groupby(['animal','CONDITION'])[['tSNE1','tSNE2']].median()


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


######################################################################################
######################## SAVE FOR STATS ##############################################
######################################################################################

#sel = (df_estimators_iso.decay_isol < 10)&(df_estimators_iso.decay_isol > 1/30.) #,'height.median_norm',  'decay_isol_norm'
#df_for_stats_iso = df_estimators_iso[['time','condition','animal','n_peaks_norm','height.median_norm','decay_isol_norm']][sel].groupby(['time', 'animal','condition'], as_index=False).mean()
#df_for_stats_iso.loc[:,'condition'] = 'iso'

#sel = (df_estimators_keta.decay_isol < 10)&(df_estimators_keta.decay_isol > 1/30.) #,'height.median_norm',  'decay_isol_norm'
#df_for_stats_keta = df_estimators_keta[['time','condition','animal','n_peaks_norm','height.median_norm','decay_isol_norm']][sel].groupby(['time', 'animal','condition'], as_index=False).mean()
#df_for_stats_keta.loc[:,'condition'] = 'keta'

#sel = (df_estimators_mmf.decay_isol < 10)&(df_estimators_mmf.decay_isol > 1/30.) #,'height.median_norm',  'decay_isol_norm'
#df_for_stats_mmf = df_estimators_mmf[['time','condition','animal','n_peaks_norm','height.median_norm','decay_isol_norm']][sel].groupby(['time', 'animal','condition'], as_index=False).mean()
#df_for_stats_mmf.loc[:,'condition'] = 'mmf'

#df_for_stats = pd.concat([df_for_stats_iso,df_for_stats_keta,df_for_stats_mmf])

#df_for_stats.to_excel("./transition_state.xlsx")

######################################################################################
######################## SAVE FOR STATS ##############################################
######################################################################################


sel = (df_estimators_iso.decay_isol < 10)&(df_estimators_iso.decay_isol > 1/30.) #,'height.median_norm',  'decay_isol_norm'
df_for_stats_iso = df_estimators_iso[['time','condition','animal','neuronID','n_peaks_norm','height.median_norm','decay_isol_norm']][sel]
df_for_stats_iso.loc[:,'condition'] = 'iso'

# quick and dirty fix for neuronID

### amount of neurons in ISO
### 
### 8235 - 763
### 8237 - 1228 
### 8238 - 546
### F0 - 554
### M0 - 111

#df_for_stats_iso[ df_for_stats_iso.animal == 'TS_C_ISO_8235']['neuronID'] 

df_for_stats_iso.loc[df_for_stats_iso.animal == 'TS_C_ISO_8235','neuronID'] = df_for_stats_iso.loc[df_for_stats_iso.animal == 'TS_C_ISO_8235','neuronID']
df_for_stats_iso.loc[df_for_stats_iso.animal == 'TS_C_ISO_8237','neuronID'] = 763 + df_for_stats_iso.loc[df_for_stats_iso.animal == 'TS_C_ISO_8237','neuronID']
df_for_stats_iso.loc[df_for_stats_iso.animal == 'TS_C_ISO_8238','neuronID'] = 763 + 1228 + df_for_stats_iso.loc[df_for_stats_iso.animal == 'TS_C_ISO_8238','neuronID']
df_for_stats_iso.loc[df_for_stats_iso.animal == 'TS_C_ISO_F0','neuronID'] =   763 + 1228 + 546 + df_for_stats_iso.loc[df_for_stats_iso.animal == 'TS_C_ISO_F0','neuronID']
df_for_stats_iso.loc[df_for_stats_iso.animal == 'TS_C_ISO_M0','neuronID'] =   763 + 1228 + 546 + 554 + df_for_stats_iso.loc[df_for_stats_iso.animal == 'TS_C_ISO_M0','neuronID']



sel = (df_estimators_keta.decay_isol < 10)&(df_estimators_keta.decay_isol > 1/30.) #,'height.median_norm',  'decay_isol_norm'
df_for_stats_keta = df_estimators_keta[['time','condition','animal','neuronID','n_peaks_norm','height.median_norm','decay_isol_norm']][sel]
df_for_stats_keta.loc[:,'condition'] = 'keta'


df_for_stats_keta[ df_for_stats_keta.animal == 'TS_C_KETA_M3']['neuronID'] 

### amount of neurons in KETA
### 
### 8235 - 550
### 8237 - 945
### 8238 - 500
### F0 -   543
### M3 -   176

df_for_stats_keta.loc[df_for_stats_keta.animal == 'TS_C_KETA_8235','neuronID'] = df_for_stats_keta.loc[df_for_stats_keta.animal == 'TS_C_KETA_8235','neuronID']
df_for_stats_keta.loc[df_for_stats_keta.animal == 'TS_C_KETA_8237','neuronID'] = 550 + df_for_stats_keta.loc[df_for_stats_keta.animal == 'TS_C_KETA_8237','neuronID']
df_for_stats_keta.loc[df_for_stats_keta.animal == 'TS_C_KETA_8238','neuronID'] = 550 + 945 + df_for_stats_keta.loc[df_for_stats_keta.animal == 'TS_C_KETA_8238','neuronID']
df_for_stats_keta.loc[df_for_stats_keta.animal == 'TS_C_KETA_F0','neuronID'] =   550 + 945 + 500 + df_for_stats_keta.loc[df_for_stats_keta.animal == 'TS_C_KETA_F0','neuronID']
df_for_stats_keta.loc[df_for_stats_keta.animal == 'TS_C_KETA_M3','neuronID'] =   550 + 945 + 500 + 543 + df_for_stats_keta.loc[df_for_stats_keta.animal == 'TS_C_KETA_M3','neuronID']



sel = (df_estimators_mmf.decay_isol < 10)&(df_estimators_mmf.decay_isol > 1/30.) #,'height.median_norm',  'decay_isol_norm'
df_for_stats_mmf = df_estimators_mmf[['time','condition','animal','neuronID','n_peaks_norm','height.median_norm','decay_isol_norm']][sel]
df_for_stats_mmf.loc[:,'condition'] = 'mmf'


df_for_stats_mmf[ df_for_stats_mmf.animal == 'TS_C_MMF_F0']['neuronID'] 

### amount of neurons in MMF
### 
### 8235 - 583
### 8237 - 1169
### 8238 - 314
### F0 - 484
### M0 - 103  

df_for_stats_mmf.loc[df_for_stats_mmf.animal == 'TS_C_MMF_8235','neuronID'] = df_for_stats_mmf.loc[df_for_stats_mmf.animal == 'TS_C_MMF_8235','neuronID']
df_for_stats_mmf.loc[df_for_stats_mmf.animal == 'TS_C_MMF_8237','neuronID'] = 583 + df_for_stats_mmf.loc[df_for_stats_mmf.animal == 'TS_C_MMF_8237','neuronID']
df_for_stats_mmf.loc[df_for_stats_mmf.animal == 'TS_C_MMF_8238','neuronID'] = 583 + 1169 + df_for_stats_mmf.loc[df_for_stats_mmf.animal == 'TS_C_MMF_8238','neuronID']
df_for_stats_mmf.loc[df_for_stats_mmf.animal == 'TS_C_MMF_F0','neuronID'] =   583 + 1169 + 314 + df_for_stats_mmf.loc[df_for_stats_mmf.animal == 'TS_C_MMF_F0','neuronID']
df_for_stats_mmf.loc[df_for_stats_mmf.animal == 'TS_C_MMF_M3','neuronID'] =   583 + 1169 + 314 + 484 + df_for_stats_mmf.loc[df_for_stats_mmf.animal == 'TS_C_MMF_M3','neuronID']



sel = (df_estimators_awake.decay_isol < 10)&(df_estimators_awake.decay_isol > 1/30.) #,'height.median_norm',  'decay_isol_norm'
df_for_stats_awake = df_estimators_awake[['time','condition','animal','neuronID','n_peaks_norm','height.median_norm','decay_isol_norm']][sel]
df_for_stats_awake.loc[:,'condition'] = 'awake'


df_for_stats_awake[df_for_stats_awake.animal == 'TS_C_AWAKE_CTRL_8235']['neuronID'] 

### amount of neurons in Awake
### 
### F0 - 589
### 8235 - 796
### 8237 - 865
### 8238 - 464


df_for_stats_awake.loc[df_for_stats_awake.animal == 'TS_C_AWAKE_CTRL_F0','neuronID'] =   df_for_stats_awake.loc[df_for_stats_awake.animal == 'TS_C_AWAKE_CTRL_F0','neuronID']
df_for_stats_awake.loc[df_for_stats_awake.animal == 'TS_C_AWAKE_CTRL_8235','neuronID'] = 589 + df_for_stats_awake.loc[df_for_stats_awake.animal == 'TS_C_AWAKE_CTRL_8235','neuronID']
df_for_stats_awake.loc[df_for_stats_awake.animal == 'TS_C_AWAKE_CTRL_8237','neuronID'] = 589 + 796 + df_for_stats_awake.loc[df_for_stats_awake.animal == 'TS_C_AWAKE_CTRL_8237','neuronID']
df_for_stats_awake.loc[df_for_stats_awake.animal == 'TS_C_AWAKE_CTRL_8238','neuronID'] = 589 + 796 + 865 + df_for_stats_awake.loc[df_for_stats_awake.animal == 'TS_C_AWAKE_CTRL_8238','neuronID']



df_for_stats = pd.concat([df_for_stats_iso,df_for_stats_keta,df_for_stats_mmf,df_for_stats_awake])

df_for_stats.to_excel("./transition_state_same_neuron.xlsx")


#### CTRL group

#### Global indexing
### - check for neurons that is intersecting and without NaNs
















####################################################################################################
####################################################################################################
#################### MODULATION INDEX ##############################################################
####################################################################################################
####################################################################################################


### CALCULATE MODULATED N_PEAKS


df_estimators_mmf['n_peaks_mod'] = df_estimators_mmf['n_peaks']

for animal in ['TS_C_MMF_8235', 'TS_C_MMF_8237', 'TS_C_MMF_8238', 'TS_C_MMF_F0','TS_C_MMF_M0']: 
    
    sel = (df_estimators_mmf.n_peaks > 0.3)    
    sel_animal = (df_estimators_mmf.animal == animal)
    mmf_awake_median = df_estimators_mmf[sel&sel_animal&(df_estimators_mmf.condition == 'awake')]['n_peaks_mod'].dropna().median()
    yyy = df_estimators_mmf.loc[sel&sel_animal, 'n_peaks_mod' ]
    yyy =  (yyy - mmf_awake_median)/(yyy + mmf_awake_median)
    df_estimators_mmf.loc[sel&sel_animal, 'n_peaks_mod']  = yyy
    df_estimators_mmf.loc[~sel&sel_animal, 'n_peaks_mod']  = np.nan
    

df_estimators_iso['n_peaks_mod'] = df_estimators_iso['n_peaks']

for animal in ['TS_C_ISO_8235', 'TS_C_ISO_8237', 'TS_C_ISO_8238', 'TS_C_ISO_M0','TS_C_ISO_F0']: 

    sel = (df_estimators_iso.n_peaks > 0.3) 
    sel_animal = (df_estimators_iso.animal == animal)
    iso_awake_median = df_estimators_iso[sel&sel_animal&(df_estimators_iso.condition == 'awake')]['n_peaks_mod'].dropna().median()
    yyy = df_estimators_iso.loc[sel&sel_animal, 'n_peaks_mod' ]
    yyy =  (yyy - iso_awake_median)/(yyy + iso_awake_median)
    df_estimators_iso.loc[sel&sel_animal, 'n_peaks_mod']  = yyy
    df_estimators_iso.loc[~sel&sel_animal, 'n_peaks_mod']  = np.nan


df_estimators_keta['n_peaks_mod'] = df_estimators_keta['n_peaks']

for animal in ['TS_C_KETA_8235', 'TS_C_KETA_8237', 'TS_C_KETA_8238', 'TS_C_KETA_F0','TS_C_KETA_M3']: 

    sel = (df_estimators_keta.n_peaks > 0.3)
    sel_animal = (df_estimators_keta.animal == animal)
    keta_awake_median = df_estimators_keta[sel&sel_animal&(df_estimators_keta.condition == 'awake')]['n_peaks_mod'].dropna().median()
    yyy = df_estimators_keta.loc[sel&sel_animal, 'n_peaks_mod' ]
    yyy =  (yyy - keta_awake_median)/(yyy + keta_awake_median)
    df_estimators_keta.loc[sel&sel_animal, 'n_peaks_mod']  = yyy
    df_estimators_keta.loc[~sel&sel_animal, 'n_peaks_mod']  = np.nan

# ZERO PEAKES EXCLUDED

sns.lineplot(x='time', y='n_peaks_mod', markers='o',lw=3,estimator=np.median,ci=95, color=color_keta,data=df_estimators_keta.dropna(),label='Keta/Xyl')
sns.lineplot(x='time', y='n_peaks_mod', markers='o',lw=3,estimator=np.median,ci=95, color=color_mmf,data=df_estimators_mmf.dropna(),label='MMF')
sns.lineplot(x='time', y='n_peaks_mod', markers='o',lw=3,estimator=np.median,ci=95, color=color_iso,data=df_estimators_iso.dropna(),label='Iso')
#sns.lineplot(x='time', y='n_peaks_mod', markers='o',lw=3,estimator=np.mean,ci=95, color=color_awake,data=df_estimators_awake.dropna(),label='Awake')

#plt.plot([30,30],[0,2.0],'--k')
#plt.plot([90,90],[0,2.0],'--k')
plt.legend(fontsize=15)
plt.ylabel("modulation index (number of transients)",fontsize=15)
#plt.title("modulation index (transients)",fontsize=22)
plt.xlabel("Time, [min]", fontsize=15)
plt.xlim([0.0, 360.0])


plt.savefig("n_peaks_mod.svg", format = 'svg', dpi=300)
#plt.savefig('height.median_norm.eps', format='eps')
plt.savefig('n_peaks_mod.png', format='png')


################################################################################
################################################################################


### CALCULATE MODULATE AMPLITUDE

df_estimators_mmf['height.median_mod'] = df_estimators_mmf['height.median']

for animal in ['TS_C_MMF_8235', 'TS_C_MMF_8237', 'TS_C_MMF_8238', 'TS_C_MMF_F0','TS_C_MMF_M0']: 
      
    sel = (df_estimators_mmf.n_peaks > 0.3)
    sel_animal = (df_estimators_mmf.animal == animal)
    mmf_awake_median = df_estimators_mmf[sel&sel_animal&(df_estimators_mmf.condition == 'awake')]['height.median_mod'].median()
    yyy = df_estimators_mmf.loc[sel&sel_animal, 'height.median_mod' ]
    yyy =  (yyy - mmf_awake_median)/(yyy + mmf_awake_median)
    df_estimators_mmf.loc[sel&sel_animal, 'height.median_mod']  = yyy

df_estimators_iso['height.median_mod'] = df_estimators_iso['height.median']

for animal in ['TS_C_ISO_8235', 'TS_C_ISO_8237', 'TS_C_ISO_8238', 'TS_C_ISO_M0','TS_C_ISO_F0']: 

    sel = (df_estimators_iso.n_peaks > 0.3)     
    sel_animal = (df_estimators_iso.animal == animal)
    iso_awake_median = df_estimators_iso[sel&sel_animal&(df_estimators_iso.condition == 'awake')]['height.median_mod'].median()
    yyy = df_estimators_iso.loc[sel&sel_animal, 'height.median_mod' ]
    yyy =  (yyy - iso_awake_median)/(yyy + iso_awake_median)
    df_estimators_iso.loc[sel&sel_animal, 'height.median_mod']  = yyy

df_estimators_keta['height.median_mod'] = df_estimators_keta['height.median']

for animal in ['TS_C_KETA_8235', 'TS_C_KETA_8237', 'TS_C_KETA_8238', 'TS_C_KETA_F0','TS_C_KETA_M3']: 
      
    sel = (df_estimators_keta.n_peaks > 0.3)
    sel_animal = (df_estimators_keta.animal == animal)
    keta_awake_median = df_estimators_keta[sel&sel_animal&(df_estimators_keta.condition == 'awake')]['height.median_mod'].median()
    yyy = df_estimators_keta.loc[sel&sel_animal, 'height.median_mod' ]
    yyy =  (yyy - keta_awake_median)/(yyy + keta_awake_median)
    df_estimators_keta.loc[sel&sel_animal, 'height.median_mod']  = yyy


df_estimators_awake['height.median_mod'] = df_estimators_awake['height.median']


for animal in ['TS_C_AWAKE_CTRL_8235', 'TS_C_AWAKE_CTRL_8237', 'TS_C_AWAKE_CTRL_8238', 'TS_C_AWAKE_CTRL_F0']: 
      
    sel = (df_estimators_awake.n_peaks > 0.3)
    sel_animal = (df_estimators_awake.animal == animal)
    awake1_median = df_estimators_awake[sel&sel_animal&(df_estimators_awake.condition == 'awake1')]['height.median_mod'].median()
    yyy = df_estimators_awake.loc[sel&sel_animal, 'height.median_mod' ]
    yyy =  (yyy - awake1_median)/(yyy + awake1_median)
    df_estimators_awake.loc[sel&sel_animal, 'height.median_mod']  = yyy
    
    

### AMPLITUDE

sns.lineplot(x='time', y='height.median_mod', lw=3, estimator=np.median, ci=95, color=color_mmf, data=df_estimators_mmf.dropna(), label='MMF')
sns.lineplot(x='time', y='height.median_mod', markers='o',lw=3,estimator=np.median,ci=95, color=color_keta,data=df_estimators_keta.dropna(),label='Keta/Xyl')
sns.lineplot(x='time', y='height.median_mod', markers='o',lw=3,estimator=np.median,ci=95, color=color_iso,data=df_estimators_iso.dropna(),label='Iso')
#sns.lineplot(x='time', y='height.median_mod', markers='o',lw=3,estimator=np.mean,ci=95, color=color_awake,data=df_estimators_awake.dropna(),label='Awake')

#plt.plot([30,30],[0,2.0],'--k')
#plt.plot([90,90],[0,2.0],'--k')
plt.legend(fontsize=15)
#plt.title("modulation index (transients)",fontsize=22)
plt.ylabel("modulation index (amplitude)",fontsize=15)
plt.xlabel("Time, [min]", fontsize=15)
#plt.ylim([0.5,1.5])
plt.xlim([0,360])

plt.savefig("height.median_mod.svg", format = 'svg', dpi=300)
#plt.savefig('height.median_norm.eps', format='eps')
plt.savefig('height.median_mod.png', format='png')


### CALCULATE MODULATE DECAY CONSTANT


df_estimators_mmf['decay_isol_mod'] = df_estimators_mmf['decay_isol']

for animal in ['TS_C_MMF_8235', 'TS_C_MMF_8237', 'TS_C_MMF_8238', 'TS_C_MMF_F0','TS_C_MMF_M0']: 
    
    sel = (df_estimators_mmf.n_peaks > 0.3)&(df_estimators_mmf.decay_isol > 1/30.)&(df_estimators_mmf.decay_isol < 10)
    sel_animal = (df_estimators_mmf.animal == animal)
    mmf_awake_median = df_estimators_mmf[sel&sel_animal&(df_estimators_mmf.condition == 'awake')]['decay_isol_mod'].median()
    yyy = df_estimators_mmf.loc[sel&sel_animal, 'decay_isol_mod' ]
    yyy =  (yyy - mmf_awake_median)/(yyy + mmf_awake_median)
    df_estimators_mmf.loc[sel&sel_animal, 'decay_isol_mod']  = yyy
    df_estimators_mmf.loc[~sel&sel_animal, 'decay_isol_mod']  = np.nan

df_estimators_iso['decay_isol_mod'] = df_estimators_iso['decay_isol']

for animal in ['TS_C_ISO_8235', 'TS_C_ISO_8237', 'TS_C_ISO_8238', 'TS_C_ISO_M0','TS_C_ISO_F0']: 

    sel = (df_estimators_iso.n_peaks > 0.3)&(df_estimators_iso.decay_isol > 1/30.)&(df_estimators_iso.decay_isol < 10)
    sel_animal = (df_estimators_iso.animal == animal)
    iso_awake_median = df_estimators_iso[sel&sel_animal&(df_estimators_iso.condition == 'awake')]['decay_isol_mod'].median()
    yyy = df_estimators_iso.loc[sel&sel_animal, 'decay_isol_mod' ]
    yyy =  (yyy - iso_awake_median)/(yyy + iso_awake_median)
    df_estimators_iso.loc[sel&sel_animal, 'decay_isol_mod']  = yyy
    df_estimators_iso.loc[~sel&sel_animal, 'decay_isol_mod']  = np.nan

df_estimators_keta['decay_isol_mod'] = df_estimators_keta['decay_isol']

for animal in ['TS_C_KETA_8235', 'TS_C_KETA_8237', 'TS_C_KETA_8238', 'TS_C_KETA_F0','TS_C_KETA_M3']: 

    sel = (df_estimators_keta.n_peaks > 0.3)&(df_estimators_keta.decay_isol > 1/30.)&(df_estimators_keta.decay_isol < 10)
    sel_animal = (df_estimators_keta.animal == animal)
    keta_awake_median = df_estimators_keta[sel&sel_animal&(df_estimators_keta.condition == 'awake')]['decay_isol_mod'].median()
    yyy = df_estimators_keta.loc[sel&sel_animal, 'decay_isol_mod' ]
    yyy =  (yyy - keta_awake_median)/(yyy + keta_awake_median)
    df_estimators_keta.loc[sel&sel_animal, 'decay_isol_mod']  = yyy
    df_estimators_keta.loc[~sel&sel_animal, 'decay_isol_mod']  = np.nan

df_estimators_awake['decay_isol_mod'] = df_estimators_awake['decay_isol']

for animal in ['TS_C_AWAKE_CTRL_8235', 'TS_C_AWAKE_CTRL_8237', 'TS_C_AWAKE_CTRL_8238', 'TS_C_AWAKE_CTRL_F0']: 

    sel = (df_estimators_awake.n_peaks > 0.3)&(df_estimators_awake.decay_isol > 1/30.)&(df_estimators_awake.decay_isol < 10)    
    sel_animal = (df_estimators_awake.animal == animal)
    awake1_median = df_estimators_awake[sel&sel_animal&(df_estimators_awake.condition == 'awake1')]['decay_isol_mod'].median()
    yyy = df_estimators_awake.loc[sel&sel_animal, 'decay_isol_mod' ]
    yyy =  (yyy - awake1_median)/(yyy + awake1_median)
    df_estimators_awake.loc[sel&sel_animal, 'decay_isol_mod']  = yyy
    df_estimators_awake.loc[~sel&sel_animal, 'decay_isol_mod']  = np.nan



#(df_estimators_keta.n_peaks > 0.3)&(df_estimators_keta.condition == 'keta')&(df_estimators_keta.decay_isol > 1/30.)

sns.lineplot(x='time', y='decay_isol_mod', markers='o',lw=3,estimator=np.median,ci=95, color=color_mmf,data=df_estimators_mmf.dropna(),label='MMF')
sns.lineplot(x='time', y='decay_isol_mod', markers='o',lw=3,estimator=np.median,ci=95, color=color_keta,data=df_estimators_keta.dropna(),label='Keta/Xyl')
sns.lineplot(x='time', y='decay_isol_mod', markers='o',lw=3,estimator=np.median,ci=95, color=color_iso,data=df_estimators_iso.dropna(),label='Iso')
#sns.lineplot(x='time', y='decay_isol_mod', markers='o',lw=3,estimator=np.median,ci=95, color=color_awake,data=df_estimators_awake.dropna(),label='Awake')

#plt.plot([30,30],[0,2.0],'--k')
#plt.plot([90,90],[0,2.0],'--k')

plt.xlim([0,360])
plt.legend(fontsize=15)
#plt.title("relative decay constant",fontsize=22)
plt.ylabel("modulation index (decay)",fontsize=15)
plt.xlabel("time (min)", fontsize=15)
#plt.ylim([0.5,1.5])

plt.savefig("decay_isol_mod.svg", format = 'svg', dpi=300)
#plt.savefig('height.median_norm.eps', format='eps')
plt.savefig('decay_isol_mod.png', format='png')


###########################################################################
###########################################################################


df_for_stats = pd.concat([df_estimators_iso,df_estimators_keta,df_estimators_mmf])

df_for_stats[['time','condition','animal','neuronID','n_peaks_mod','height.median_mod','decay_isol_mod']].to_excel("./transition_state_same_neuron.xlsx")



test = pd.read_excel("./transition_state_same_neuron.xlsx")

test[(test.animal == 'TS_C_ISO_8238')&(test.time == 0)].median()

sns.lineplot(x='time', y='height.median_mod', lw=3, estimator=np.median, ci=95, color=color_mmf, data=test[(test.animal == 'TS_C_MMF_8235')|(test.animal == 'TS_C_MMF_8237')|(test.animal == 'TS_C_MMF_8238')|(test.animal == 'TS_C_MMF_F0')].dropna(), label='MMF')



sns.lineplot(x='time', y='height.median_mod', markers='o',lw=3,estimator=np.median,ci=95, color=color_keta,data=test[].dropna(),label='Keta/Xyl')
sns.lineplot(x='time', y='height.median_mod', markers='o',lw=3,estimator=np.median,ci=95, color=color_iso,data=test.dropna(),label='Iso')
#sns.lineplot(x='time', y='height.median_mod', markers='o',lw=3,estimator=np.mean,ci=95, color=color_awake,data=df_estimators_awake.dropna(),label='Awake')





#############################################################################
#############################################################################


df_for_stats_iso = df_estimators_iso
df_for_stats_iso.loc[:,'condition'] = 'iso'


n_iso_8235 = max(df_estimators_iso[ df_estimators_iso.animal == 'TS_C_ISO_8235']['neuronID']) + 1 
n_iso_8237 = max(df_estimators_iso[ df_estimators_iso.animal == 'TS_C_ISO_8237']['neuronID']) + 1
n_iso_8238 = max(df_estimators_iso[ df_estimators_iso.animal == 'TS_C_ISO_8238']['neuronID']) + 1
n_iso_F0 = max(df_estimators_iso[ df_estimators_iso.animal == 'TS_C_ISO_F0']['neuronID']) + 1

df_estimators_iso.loc[df_estimators_iso.animal == 'TS_C_ISO_8235','neuronID'] = df_estimators_iso.loc[df_estimators_iso.animal == 'TS_C_ISO_8235','neuronID']
df_estimators_iso.loc[df_estimators_iso.animal == 'TS_C_ISO_8237','neuronID'] = n_iso_8235  + df_estimators_iso.loc[df_estimators_iso.animal == 'TS_C_ISO_8237','neuronID']
df_estimators_iso.loc[df_estimators_iso.animal == 'TS_C_ISO_8238','neuronID'] = n_iso_8235  + n_iso_8237 + df_estimators_iso.loc[df_estimators_iso.animal == 'TS_C_ISO_8238','neuronID']
df_estimators_iso.loc[df_estimators_iso.animal == 'TS_C_ISO_F0','neuronID'] =  n_iso_8235 + n_iso_8237 + n_iso_8238 + df_estimators_iso.loc[df_estimators_iso.animal == 'TS_C_ISO_F0','neuronID']
df_estimators_iso.loc[df_estimators_iso.animal == 'TS_C_ISO_M0','neuronID'] =   n_iso_8235 + n_iso_8237 + n_iso_8238 + n_iso_F0 + df_estimators_iso.loc[df_estimators_iso.animal == 'TS_C_ISO_M0','neuronID']











































######### BUGGY CODE #####################################################
##########################################################################


# quick and dirty fix for neuronID

#(df_estimators_awake.n_peaks > 0.3)&(df_estimators_awake.decay_isol > 1/30.)


df_for_stats_iso = df_estimators_iso
df_for_stats_iso.loc[:,'condition'] = 'iso'


n_iso_8235 = max(df_for_stats_iso[ df_for_stats_iso.animal == 'TS_C_ISO_8235']['neuronID']) + 1 
n_iso_8237 = max(df_for_stats_iso[ df_for_stats_iso.animal == 'TS_C_ISO_8237']['neuronID']) + 1
n_iso_8238 = max(df_for_stats_iso[ df_for_stats_iso.animal == 'TS_C_ISO_8238']['neuronID']) + 1
n_iso_F0 = max(df_for_stats_iso[ df_for_stats_iso.animal == 'TS_C_ISO_F0']['neuronID']) + 1

df_for_stats_iso.loc[df_for_stats_iso.animal == 'TS_C_ISO_8235','neuronID'] = df_for_stats_iso.loc[df_for_stats_iso.animal == 'TS_C_ISO_8235','neuronID']
df_for_stats_iso.loc[df_for_stats_iso.animal == 'TS_C_ISO_8237','neuronID'] = n_iso_8235  + df_for_stats_iso.loc[df_for_stats_iso.animal == 'TS_C_ISO_8237','neuronID']
df_for_stats_iso.loc[df_for_stats_iso.animal == 'TS_C_ISO_8238','neuronID'] = n_iso_8235  + n_iso_8237 + df_for_stats_iso.loc[df_for_stats_iso.animal == 'TS_C_ISO_8238','neuronID']
df_for_stats_iso.loc[df_for_stats_iso.animal == 'TS_C_ISO_F0','neuronID'] =  n_iso_8235 + n_iso_8237 + n_iso_8238 + df_for_stats_iso.loc[df_for_stats_iso.animal == 'TS_C_ISO_F0','neuronID']
df_for_stats_iso.loc[df_for_stats_iso.animal == 'TS_C_ISO_M0','neuronID'] =   n_iso_8235 + n_iso_8237 + n_iso_8238 + n_iso_F0 + df_for_stats_iso.loc[df_for_stats_iso.animal == 'TS_C_ISO_M0','neuronID']




df_for_stats_keta = df_estimators_keta
df_for_stats_keta.loc[:,'condition'] = 'keta'


df_for_stats_keta[ df_for_stats_keta.animal == 'TS_C_KETA_M3']['neuronID'] 


n_keta_8235 = max(df_for_stats_keta[ df_for_stats_keta.animal == 'TS_C_KETA_8235']['neuronID']) + 1 
n_keta_8237 = max(df_for_stats_keta[ df_for_stats_keta.animal == 'TS_C_KETA_8237']['neuronID']) + 1
n_keta_8238 = max(df_for_stats_keta[ df_for_stats_keta.animal == 'TS_C_KETA_8238']['neuronID']) + 1
n_keta_F0 = max(df_for_stats_keta[ df_for_stats_keta.animal == 'TS_C_KETA_F0']['neuronID']) + 1

df_for_stats_keta.loc[df_for_stats_keta.animal == 'TS_C_KETA_8235','neuronID'] = df_for_stats_keta.loc[df_for_stats_keta.animal == 'TS_C_KETA_8235','neuronID']
df_for_stats_keta.loc[df_for_stats_keta.animal == 'TS_C_KETA_8237','neuronID'] = n_keta_8235 + df_for_stats_keta.loc[df_for_stats_keta.animal == 'TS_C_KETA_8237','neuronID']
df_for_stats_keta.loc[df_for_stats_keta.animal == 'TS_C_KETA_8238','neuronID'] = n_keta_8235 + n_keta_8237 + df_for_stats_keta.loc[df_for_stats_keta.animal == 'TS_C_KETA_8238','neuronID']
df_for_stats_keta.loc[df_for_stats_keta.animal == 'TS_C_KETA_F0','neuronID'] =   n_keta_8235 + n_keta_8237 + n_keta_8238 + df_for_stats_keta.loc[df_for_stats_keta.animal == 'TS_C_KETA_F0','neuronID']
df_for_stats_keta.loc[df_for_stats_keta.animal == 'TS_C_KETA_M3','neuronID'] =   n_keta_8235 + n_keta_8237 + n_keta_8238 + n_keta_F0 + df_for_stats_keta.loc[df_for_stats_keta.animal == 'TS_C_KETA_M3','neuronID']



df_for_stats_mmf = df_estimators_mmf
df_for_stats_mmf.loc[:,'condition'] = 'mmf'


df_for_stats_mmf[ df_for_stats_mmf.animal == 'TS_C_MMF_F0']['neuronID'] 
  


n_mmf_8235 = max(df_for_stats_mmf[ df_for_stats_mmf.animal == 'TS_C_MMF_8235']['neuronID']) + 1 
n_mmf_8237 = max(df_for_stats_mmf[ df_for_stats_mmf.animal == 'TS_C_MMF_8237']['neuronID']) + 1
n_mmf_8238 = max(df_for_stats_mmf[ df_for_stats_mmf.animal == 'TS_C_MMF_8238']['neuronID']) + 1
n_mmf_F0 = max(df_for_stats_mmf[ df_for_stats_mmf.animal == 'TS_C_MMF_F0']['neuronID']) + 1                                                                       
                                                                               
                                                                              

df_for_stats_mmf.loc[df_for_stats_mmf.animal == 'TS_C_MMF_8235','neuronID'] = df_for_stats_mmf.loc[df_for_stats_mmf.animal == 'TS_C_MMF_8235','neuronID']
df_for_stats_mmf.loc[df_for_stats_mmf.animal == 'TS_C_MMF_8237','neuronID'] = n_mmf_8235 + df_for_stats_mmf.loc[df_for_stats_mmf.animal == 'TS_C_MMF_8237','neuronID']
df_for_stats_mmf.loc[df_for_stats_mmf.animal == 'TS_C_MMF_8238','neuronID'] = n_mmf_8235 + n_mmf_8237 + df_for_stats_mmf.loc[df_for_stats_mmf.animal == 'TS_C_MMF_8238','neuronID']
df_for_stats_mmf.loc[df_for_stats_mmf.animal == 'TS_C_MMF_F0','neuronID'] =   n_mmf_8235 + n_mmf_8237 + n_mmf_8238 + df_for_stats_mmf.loc[df_for_stats_mmf.animal == 'TS_C_MMF_F0','neuronID']
df_for_stats_mmf.loc[df_for_stats_mmf.animal == 'TS_C_MMF_M3','neuronID'] =   n_mmf_8235 + n_mmf_8237 + n_mmf_8238 + n_mmf_F0 + df_for_stats_mmf.loc[df_for_stats_mmf.animal == 'TS_C_MMF_M3','neuronID']



df_for_stats = pd.concat([df_for_stats_iso,df_for_stats_keta,df_for_stats_mmf])


##### Plot check for stats



sns.lineplot(x='time', y='n_peaks_mod', markers='o',lw=3, estimator=np.median,ci=95, color=color_mmf,data=df_for_stats[df_for_stats.condition == 'mmf'].dropna(),label='MMF')
sns.lineplot(x='time', y='n_peaks_mod', markers='o',lw=3, estimator=np.median,ci=95, color=color_keta,data=df_for_stats[df_for_stats.condition == 'keta'].dropna(),label='Keta/Xyl')
sns.lineplot(x='time', y='n_peaks_mod', markers='o',lw=3, estimator=np.median,ci=95, color=color_iso,data=df_for_stats[df_for_stats.condition == 'iso'].dropna(),label='Iso')
sns.lineplot(x='time', y='n_peaks_mod', markers='o',lw=3, estimator=np.median,ci=95, color=color_awake,data=df_for_stats[df_for_stats.condition == 'awake'].dropna(),label='Awake')


sns.lineplot(x='time', y='height.median_mod', markers='o',lw=3, estimator=np.median,ci=95, color=color_mmf,data=df_for_stats[df_for_stats.condition == 'mmf'].dropna(),label='MMF')
sns.lineplot(x='time', y='height.median_mod', markers='o',lw=3, estimator=np.median,ci=95, color=color_keta,data=df_for_stats[df_for_stats.condition == 'keta'].dropna(),label='Keta/Xyl')
sns.lineplot(x='time', y='height.median_mod', markers='o',lw=3, estimator=np.median,ci=95, color=color_iso,data=df_for_stats[df_for_stats.condition == 'iso'].dropna(),label='Iso')
sns.lineplot(x='time', y='height.median_mod', markers='o',lw=3, estimator=np.median,ci=95, color=color_awake,data=df_for_stats[df_for_stats.condition == 'awake'].dropna(),label='Awake')


sns.lineplot(x='time', y='decay_isol_mod', markers='o',lw=3,estimator=np.median,ci=95, color=color_mmf,data=df_for_stats[df_for_stats.condition == 'mmf'].dropna(),label='MMF')
sns.lineplot(x='time', y='decay_isol_mod', markers='o',lw=3,estimator=np.median,ci=95, color=color_keta,data=df_for_stats[df_for_stats.condition == 'keta'].dropna(),label='Keta/Xyl')
sns.lineplot(x='time', y='decay_isol_mod', markers='o',lw=3,estimator=np.median,ci=95, color=color_iso,data=df_for_stats[df_for_stats.condition == 'iso'].dropna(),label='Iso')
sns.lineplot(x='time', y='decay_isol_mod', markers='o',lw=3,estimator=np.median,ci=95, color=color_awake,data=df_for_stats[df_for_stats.condition == 'awake'].dropna(),label='Awake')


df_for_stats.to_excel("/media/andrey/My Passport/Yang_Chini_et_al/Stats (R)/scripts/transition_state_modulation_index_same_neuron.xlsx")


sns.lineplot(x='time', y='height.median_mod', markers='o',lw=3, estimator=np.mean,ci=95, color=color_mmf,data=df_for_stats[df_for_stats.condition == 'mmf'].dropna(),label='MMF')
sns.lineplot(x='time', y='height.median_mod', markers='o',lw=3, estimator=np.mean,ci=95, color=color_keta,data=df_for_stats[df_for_stats.condition == 'keta'].dropna(),label='Keta/Xyl')
sns.lineplot(x='time', y='height.median_mod', markers='o',lw=3, estimator=np.mean,ci=95, color=color_iso,data=df_for_stats[df_for_stats.condition == 'iso'].dropna(),label='Iso')
sns.lineplot(x='time', y='height.median_mod', markers='o',lw=3, estimator=np.mean,ci=95, color=color_awake,data=df_for_stats[df_for_stats.condition == 'awake'].dropna(),label='Awake')


















