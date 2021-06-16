import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pickle
import os

sns.set()
sns.set_style("whitegrid")

from scipy.signal import medfilt 
from scipy.stats import skew, kurtosis, zscore

from scipy import signal

from sklearn.linear_model import LinearRegression, TheilSenRegressor

from sys import path
### set path to where OASIS is located
### make sure to run ' python setup.py build_ext --inplace ' in OASIS-master folder
path.append(r'/media/andrey/My Passport/OASIS-master')
from oasis.functions import deconvolve
from oasis.functions import estimate_time_constant


def get_recordings_for_animals(animals, path):
    
    recordings = []
        
    for animal in animals:
        meta_data = pd.read_excel(path)
        meta_animal = meta_data[meta_data['Mouse'] == animal]
        recs = meta_animal['Number'].to_list()
        for r in recs:
            recordings.append(r)
        
    return recordings

def get_animal_from_recording(recording, path):
    meta_data = pd.read_excel(path)
    meta_recording = meta_data[meta_data['Number'] == recording]
    animal = (meta_recording['Mouse'])
    animal = animal.to_numpy()[0]
    return animal

def get_condition(recording, path):
    meta_data = pd.read_excel(path)
    condition = meta_data['Condition'][meta_data['Number'] == recording].values[0]
    return condition


def traces_and_npils(recording, path, concatenation=True):
    
    meta_data = pd.read_excel(path)
    
    if (concatenation==True):
        
        path_excel_rec = str(meta_data['Folder'][recording] + meta_data['Subfolder'][recording] + 'suite2p/plane0')
        
  
        Traces = np.load(path_excel_rec + '/F.npy',allow_pickle=True)
        Npil = np.load(path_excel_rec + '/Fneu.npy',allow_pickle=True)
        iscell = np.load(path_excel_rec + '/iscell.npy',allow_pickle=True)
        
        print("Total trace length: " + str(Traces.shape[1]))
        
        starting_frame = int(meta_data['Starting frame'][recording])
        recording_length = int(meta_data['Recording length'][recording])
        
        #analysis_period = meta_data['Analysis period'][recording]
        #analysis_period = [int(s) for s in analysis_period.split(',')]
        
        n_accepted_rejected = Traces.shape[0] 
        
        #print("Period for the analysis (absolute): " + str(analysis_period[0]+starting_frame)+" "+str(analysis_period[1]+starting_frame))
        
        print("Recording length: " + str(recording_length))
        
        #print("Period for the analysis (relative): " + str(analysis_period[0])+" "+str(analysis_period[1]))
        
        good_recording = np.zeros(shape=(recording_length))
        
        
        
        if isinstance(meta_data['Analysis period'][recording], str): # as previous script
            #good_recording = np.zeros((18000))
            recording2keep = [int(s) for s in meta_data['Analysis period'][recording].split(',')]
            print("Analysis periods: " + str(recording2keep))
            begin = recording2keep[0::2]
            ending = recording2keep[1::2]
            for idx_begin in range(int(len(begin))):
                good_recording[begin[idx_begin] : ending[idx_begin]] = 1
        #else:
        #    good_recording = np.ones_like(Spikes[0, :])
        
        good_recording = good_recording > 0  
        
    
        
        Traces = Traces[:,starting_frame:starting_frame+recording_length]
        Npil = Npil[:,starting_frame:starting_frame+recording_length]
        
   
        
        Traces = Traces[:, good_recording]     
        Npil = Npil[:, good_recording]

        print("Analysis period total frames: ", Traces.shape[1])
        
        #Traces = Traces[:,analysis_period[0]+starting_frame:analysis_period[1]+starting_frame]
        #Npil = Npil[:,analysis_period[0]+starting_frame:analysis_period[1]+starting_frame]

        Traces = Traces[iscell[:, 0].astype(bool), :]
        Npil = Npil[iscell[:, 0].astype(bool), :] 
         
    else:
        
        path_excel_rec = str(meta_data['Folder'][recording] + meta_data['Subfolder'][recording] +
                    str(int(meta_data['Recording idx'][recording])) + '/suite2p/plane0')
    

        Traces = np.load(path_excel_rec + '/F.npy',allow_pickle=True)
        Npil = np.load(path_excel_rec + '/Fneu.npy',allow_pickle=True)
        iscell = np.load(path_excel_rec + '/iscell.npy',allow_pickle=True)
        
        n_accepted_rejected = Traces.shape[0]
        
        Traces = Traces[iscell[:, 0].astype(bool), :]
        Npil = Npil[iscell[:, 0].astype(bool), :]   

    print("fuction: n_accepted_rejected",n_accepted_rejected)   
    return Traces, Npil, n_accepted_rejected # n_accepted_rejected = Accepted + Rejected

def median_stabilities(Npils):

    number_of_neurons = Npils.shape[0] 
    length = Npils.shape[1]
    
    #print(length)
    
    l = int(length / 1000)

    Npils = Npils[:,:l*1000]
    
    npils_medians = ma.median(Npils, axis=1)
    
    #Npils = Npils - np.tile(ma.expand_dims(ma.median(Npils, axis=1), axis=1),
    #                              (1, ma.shape(Npils)[1]))
    
    Npils = Npils.reshape(Npils.shape[0],l,1000)
    
    #npils_medians = npils_medians.reshape(Npils.shape[0],l)
    #print(Npils.shape)
    
    ###TODO npils_median
    
    median_stabilities = ma.median(Npils,axis=2)
    median_for_all_trace = ma.median(Npils,axis=[1,2])

    median_stabilities = ma.abs(median_stabilities-median_for_all_trace[:,np.newaxis])

    median_stabilities = ma.sum(median_stabilities,axis=1)/l

    return median_stabilities

def get_data_frame(recording, path, threshold=200, baseline_correction=True, concatenation=True, correlations = True):
    
    df_estimators = pd.DataFrame()

    df_corr =  pd.DataFrame()

    r = recording
    
    animal = get_animal_from_recording(r, path)
        
    condition = get_condition(r, path)
    
    print(str(r)+" "+str(animal)+" "+str(condition))
    
    plt.cla()
    
    Traces, Npils, n_accepted_and_rejected = traces_and_npils(r, path, concatenation)

    Tm0p7N = Traces - 0.7*Npils
    
    if (baseline_correction==True):
    
        #median of all traces
        med_of_all_traces = np.median(Tm0p7N,axis=0)

        plt.plot(med_of_all_traces,'k')

        #filtering 
        Tm0p7N_midfilt = medfilt(med_of_all_traces,31)

        plt.plot( Tm0p7N_midfilt, 'b')

        #regression
        TSreg = TheilSenRegressor(random_state=42)
        x = np.arange(Tm0p7N.shape[1])
        X = x[:, np.newaxis]
        fit = TSreg.fit(X, Tm0p7N_midfilt) 
                #print(fit.get_params())
        y_pred =  TSreg.predict(X)


        plt.plot(  y_pred, 'w')

        #subtract
        y_pred =  y_pred - y_pred[0]
        
        ab,resid,_,_,_ = ma.polyfit(x, y_pred, 1,full = True)    

        #print(ab,resid)    

        Tm0p7N[:,:] -= y_pred[np.newaxis, :]


        plt.title("median for all traces in  recording")
        plt.show()
  
    recording_lenght = Traces.shape[1]
    
    # old baseline, worked well 
    
    baseline = np.quantile(Tm0p7N,0.25,axis=1)
    
    print("Median baseline: {:.2f}".format(np.median(baseline)))
    
    Baseline_subtracted = Tm0p7N.copy() 
    
    for i in range(Baseline_subtracted.shape[0]):
        Baseline_subtracted[i,:] -= baseline[i]
    
    integral = Baseline_subtracted.sum(axis=1)/recording_lenght
    
    
    #print(r)
    #Npil_regression = np.polyfit(np.arange(Npil.shape[1]),Npil,1)
    
    #Npil_stability[neuron_id] = Npil_regression[0]*9000/Npil_regression[1]

    #for npil in Npil:
    #    print(npil.shape)
 
    
    n_accepted = Traces.shape[0]

    print("n_accepted ",n_accepted )  
    
    neuronID = ma.arange(n_accepted)

    print("neuronID",neuronID)  
    
    n_accepted_and_rejected = n_accepted_and_rejected

    print("n_accepted_and_rejected",n_accepted_and_rejected)  
    
    Traces_median = ma.median(Traces, axis=1) 
    Npils_median = ma.median(Npils, axis=1) 
    
    Tm0p7N_median = ma.median(Tm0p7N, axis=1)   
    
    Traces_std = ma.std(Npils, axis=1)    
    Npils_std = ma.std(Npils, axis=1)
    Tm0p7N_std = ma.std(Tm0p7N, axis=1) 
    
    Traces_mins = ma.min(Traces, axis=1)
    Traces_maxs = ma.max(Traces, axis=1)
    Traces_peak_to_peak = Traces_maxs - Traces_mins
    
    Npils_mins = ma.min(Npils, axis=1)
    Npils_maxs = ma.max(Npils, axis=1)
    Npils_peak_to_peak = Npils_maxs - Npils_mins
    
    #median subtraction
    #Traces = Traces - np.tile(ma.expand_dims(ma.median(Traces, axis=1), axis=1),
     #                             (1, ma.shape(Traces)[1]))
    #Npil = Npil - np.tile(ma.expand_dims(ma.median(Npil, axis=1), axis=1),
      #                            (1, ma.shape(Npil)[1]))
    
    Traces_skewness = skew(Traces,axis=1)
    Npils_skewness = skew(Npils,axis=1)
    
    Tm0p7N_skewness = skew(Tm0p7N,axis=1)
    
    
    
    Traces_kurtosis = kurtosis(Traces, axis=1)
    Npils_kurtosis = kurtosis(Npils, axis=1)    

    
    Npils_median_stabilities = median_stabilities(Npils)
    
    slope = ma.zeros(Npils.shape[0])
    intercept = ma.zeros(Npils.shape[0])
    residuals = ma.zeros(Npils.shape[0])

    if correlations: 
	    #Replace with smarter solution to take into account correlations with other neurons.
	    Tcorr = np.corrcoef(Traces).flatten()
	    Ncorr = np.corrcoef(Npils).flatten()
	    
	    Tm0p7Ncorr_mean = np.mean(np.corrcoef(Tm0p7N),axis=1)
	    
	    Tm0p7Ncorr = np.corrcoef(Tm0p7N).flatten()
	    
	    Tm0p7Ncorr[Tm0p7Ncorr>0.99999] = np.nan
	    
	    Tm0p7Ncorr_first100 = np.corrcoef(Tm0p7N[:100,:]).flatten()
	    
	    Tm0p7Ncorr_first100 [Tm0p7Ncorr_first100>0.99999] = np.nan
	       
	        #quick trick
	    
	    #print(Tm0p7N.shape)
	    
	    #Tm0p7N_10bins = Tm0p7N.reshape(Tm0p7N.shape[0],int(Tm0p7N.shape[1]/10),10).mean(axis=2) 
	    
	   # print(Tm0p7N_10bins.shape)
	    
	    #Tm0p7Ncorr_10bins = np.corrcoef(Tm0p7N_10bins).flatten()
	    
	   # Tm0p7Ncorr_10bins[Tm0p7Ncorr_10bins>0.99999] = np.nan
	
	    
	    
	    df_corr =  pd.DataFrame({ "animal":animal,
	                        "recording":r,
	                        "condition":condition,
	                        #"Tcorr":Tcorr,
	                        #"Ncorr":Ncorr,
	                        #"Tm0p7Ncorr":Tm0p7Ncorr,
	                        #"Tm0p7Ncorr.abs":np.absolute(Tm0p7Ncorr)
	                             
	                        "Tm0p7Ncorr":Tm0p7Ncorr,
	                        "Tm0p7Ncorr.abs":np.absolute(Tm0p7Ncorr)  
	                          #          "Tm0p7Ncorr_10bins":Tm0p7Ncorr_10bins,
	                        #"Tm0p7Ncorr_10bins.abs":np.absolute(Tm0p7Ncorr_10bins) 
	                            
                            })
                             
    i=0
    for npil in Npils: 
        ab,resid,_,_,_ = ma.polyfit(np.arange(npil.shape[0]), npil, 1,full = True)    
        slope[i] = ab[0]
        intercept[i] = ab[1]
       
        residuals[i] = resid
        i=i+1
        
    slope_per_median = ma.divide(slope,Npils_median)    
    slope_in_percent = ma.divide(ma.multiply(slope,Npils.shape[1]),Npils_median)
    
    print("Number of neurons accepted: " + str(Npils_std.shape[0]))
    
    ### decay constant and peak characterization
    
    num_cells = np.shape(Traces)[0]
    decay_isol = np.zeros((num_cells))
    decay_no_isol = np.zeros((num_cells))
    n_peaks = np.zeros((num_cells))
    height = np.zeros((num_cells))
    width = np.zeros((num_cells))
    
    baseline_oasis = np.zeros((num_cells))
    
    Baseline_subtracted = Tm0p7N.copy() 
    
    integral = np.zeros((num_cells))

    #from oasis.functions import deconvolve
    #from oasis.functions import estimate_time_constant
    
    for neuron in range(num_cells):
       
        fs=30
        _, _, b, decay_neuron_isolated10, _ = deconvolve(np.double(Tm0p7N[neuron, ]),
                                                                 penalty = 0, sn=25, optimize_g = 10)
        _, _, _, decay_neuron_no_isolated, _ = deconvolve(np.double(Tm0p7N[neuron, ]), sn=25,
                                                              penalty = 0)
        
        baseline_oasis[neuron] = b
        
        Baseline_subtracted[neuron] -= b
        
        integral[neuron] = Baseline_subtracted[neuron].sum()/recording_lenght
        #ARcoef = estimate_time_constant(Tm0p7N[neuron, ], p=2, sn=None, lags=10, fudge_factor=1.0, nonlinear_fit=False)
        
        #print(ARcoef)
    
        peak_ind, peaks = signal.find_peaks(Baseline_subtracted[neuron, ], height = threshold,
                                         distance = 10, prominence = threshold,
                                         width = (None, None),
                                         rel_height = 0.9)
        
        decay_isol[neuron] = - 1 / (fs * np.log(decay_neuron_isolated10))
        decay_no_isol[neuron] = - 1 / (fs * np.log(decay_neuron_no_isolated))
        
        if decay_isol[neuron]>0:
            pass
        else:
            decay_isol[neuron]== np.nan
            
        
    
        #print(decay_neuron_isolated10,decay_isol[neuron]," s")

        fs = 30
        trace_len = np.shape(Traces)[1] / (fs * 60) # in minutes  
        
        n_peaks[neuron] = len(peaks['peak_heights']) / trace_len
    
        if n_peaks[neuron] > 0:
            height[neuron, ] = np.median(peaks['peak_heights'])
            width[neuron, ] = np.median(peaks['widths'])
        else:
            height[neuron, ] = np.nan
            width[neuron, ] = np.nan
    

    df_estimators = pd.DataFrame({ "animal":animal,
                        "recording":r,
                        "condition":condition,
                        "neuronID":neuronID,
                        "n.accepted":n_accepted,
                        "length.frames":recording_lenght,
                        "length.minutes":trace_len,
                        "n.accepted_and_rejected":n_accepted_and_rejected,
                        "traces.median":Traces_median,
                        "npil.median":Npils_median,
                        "trace.std":Traces_std,
                        "npil.std":Npils_std,
    
                        "trace.mins":Traces_mins,
                        "trace.maxs":Traces_maxs,
                        "trace.peak_to_peak":Traces_peak_to_peak,
    
                        "npil.mins":Npils_mins,
                        "npil.maxs":Npils_maxs,
                        "npil.peak_to_peak":Npils_peak_to_peak,
    
                        "trace.skewness":Traces_skewness,
                        "npil.skewness":Npils_skewness,
                       
                        "Tm0p7N.skewness":Tm0p7N_skewness,
                        "Tm0p7N.median":Tm0p7N_median,
                        "Tm0p7N.std":Tm0p7N_std,
    
                        "trace.kurtosis":Traces_kurtosis,
                        "npil.kurtosis": Npils_kurtosis, 
    
                        "npil.slope":slope,
                        "npil.intercept":intercept,
                        "npil.residual":residuals,
                       
                        "npil.slope_per_median":slope_in_percent, 
                        "npil.slope_in_percent":slope_in_percent,
     
                        "npil.mstab.1000":Npils_median_stabilities,
                      
                        "baseline.quantile.25":baseline,
                        "baseline.oasis":baseline_oasis,
                        "integral":integral,
                       
                        #"Tm0p7Ncorr.mean":Tm0p7Ncorr_mean,
                       
                           ### decay constant and peak characterization
                        "peak_detection_threshold":threshold,
                        "decay_isol":decay_isol,
                        "decay_no_isol":decay_no_isol,
                        "n_peaks":n_peaks,
                        "n_peaks_per_recording":n_peaks*trace_len,
                        "height.median":height,
                        "width.median":width
                       
                      })
    
    return  df_estimators ,df_corr

def get_raster(starting_recording, n_neurons_max,database_path, concatenation):
    
    traces, npil, number_of_neruons = traces_and_npils(starting_recording, database_path, concatenation)
                                                       
    Tm0p7N = traces - 0.7*npil

    Tm0p7N = zscore(Tm0p7N,axis=1)   
    Tm0p7N = np.reshape(Tm0p7N,(Tm0p7N.shape[0], 500,10)).mean(axis=2)
    Tm0p7N = np.maximum(-4, np.minimum(8,  Tm0p7N)) + 4
    Tm0p7N/= 12
    
    return Tm0p7N[:n_neurons_max,:]