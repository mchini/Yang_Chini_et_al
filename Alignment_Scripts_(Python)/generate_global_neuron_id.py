# -*- coding: utf-8 -*-
"""
Created on Tue May 26 11:01:02 2020

@authors: mchini

The algorithm is searching for the same neuron across different recordings and assign a unique global ID.

### TODO visual inspection of identified neurons 

"""

threshold = 0.7 # min proportion of pixel that has to overlap

distance_threshold = 3  # minimum distance between centroids (in um)
scale_um_per_pixel = 0.41 # provide scale to convert scale from pixels inmicrometers

path = '/media/andrey/My Passport/Calcium-Imaging---Anesthesia/recordings_alignment_and_neuronID_assignment/meta_recordings_sample_small.xlsx'
path4results = '/home/andrey/Anesthesia_test/results/' #To store transformation matrisies
save_plots_path = '/home/andrey/Anesthesia_test/'
log_file_path = save_plots_path + 'neuron_id_logs.txt'


animals_for_analysis = [375290]

#%% generate an ID for unique cells across recordings of the same mouse

from pystackreg import StackReg

import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt

import sys
np.set_printoptions(threshold=sys.maxsize)

import time

#https://numba.pydata.org/numba-examples/examples/density_estimation/histogram/results.html
     
f2 = open(log_file_path, "a")

np.seterr(divide='ignore', invalid='ignore')

from sklearn.metrics.pairwise import euclidean_distances

meta_data = pd.read_excel(path)

recordings = meta_data['Number']

animals = animals_for_analysis

rat, dist =[], []
 
#%% generate an ID for unique cells across recordings of the same mouse

sr = StackReg(StackReg.AFFINE)  ## Affine or non rigit it doesn't matter

for animal in animals:
    tmats_loaded = np.load(path4results + 'StackReg/' + str(animal) + "_best_tmats" + '.npy')
    meta_animal = meta_data[meta_data['Mouse'] == animal]
    recordings = meta_animal['Number']
    image = np.zeros((512, 512, np.shape(meta_animal)[0]))
    image_match_neu = np.zeros((512, 512, np.shape(meta_animal)[0]))
    neuron_idx = 1
    neuron_properties = np.array([])
    for idx, recording in enumerate(recordings):
        print("Index of the recording:", str(idx))
        print("Global ID of the recording:", str(recording))

        stats = np.load(meta_data['Folder'][recording] + 
                          meta_data['Subfolder'][recording] +
                          str(int(meta_data['Recording idx'][recording])) +
                          '/suite2p/plane0/stat.npy',
                          allow_pickle=True)
        iscell = np.load(meta_data['Folder'][recording] + 
                          meta_data['Subfolder'][recording] +
                          str(int(meta_data['Recording idx'][recording])) +
                          '/suite2p/plane0/iscell.npy',
                          allow_pickle=True)
        stats = stats[iscell[:, 0].astype(bool)]
        IDneuron = np.zeros(np.shape(stats)[0])
        
        print("Number of neurons",np.shape(stats)[0])
        print("Number of neurons",np.shape(stats)[0], file=f2)
        
        if os.path.isfile(path4results + 'IDneuronsStackReg/' + str(animal) + '/' + str(idx) + 'bis.npy'):
            myfile = np.load(path4results + 'IDneuronsStackReg/' + str(animal) + '/' + str(idx) + 'bis.npy')
            print("File already exists:", path4results + 'IDneuronsStackReg/' + str(animal) + '/' + str(idx) + 'bis.npy')
            print("File already exists:", path4results + 'IDneuronsStackReg/' + str(animal) + '/' + str(idx) + 'bis.npy', file=f2)
            print("It was already reprocessed, so I can skip it.")
            print("It was already reprocessed, so I can skip it.", file=f2)
            last_neuron_idx = int(max(myfile))
            print("The neuron id for recording", idx, " is ", last_neuron_idx)
            print("The neuron id for recording", idx, " is ", last_neuron_idx, file=f2)
            neuron_idx = last_neuron_idx
            #print("I need only calculate neuron properties")
            #print("I need only calculate neuron properties", file=f2)
            if idx == 0:
                for stats_neuron in stats:
                    image[stats_neuron['ypix'], stats_neuron['xpix'], idx] = neuron_idx
                    neuron_idx = neuron_idx + 1
                    centroid = np.expand_dims(np.array(stats_neuron['med']), axis = 1)
                    #print(centroid)
                    neuron_properties = np.concatenate((neuron_properties, centroid), 
                                                       axis = 1) if len(neuron_properties) > 0 else centroid
                            
            else:
                for idx_neuron, stats_neuron in enumerate(stats):
                    image[stats_neuron['ypix'], stats_neuron['xpix'], idx] = myfile[idx_neuron]   
                    centroid = np.expand_dims(np.array(stats_neuron['med']), axis=1) ###TODO
                    neuron_properties = np.concatenate((neuron_properties, centroid), axis = 1)
        
            continue

        #(idx==2):
            #pass
            #print(len(rat))
            #plt.plot(rat,dist,"*")
            #plt.ylim(-5,30)
            #file_title = "_" + str(int(idx)) + "_ROI_match"
            #plt.savefig(save_plots_path + "StackRegVisualInspection/" + file_title + ".png")
            #print("saved")
            #plt.show()
            #sys.exit()
        if idx == 0:
            for stats_neuron in stats:
                image[stats_neuron['ypix'], stats_neuron['xpix'], idx] = neuron_idx ###STRANGE!!! #TODO!!!

                IDneuron[neuron_idx - 1] = neuron_idx
                neuron_idx = neuron_idx + 1
                centroid = np.expand_dims(np.array(stats_neuron['med']), axis = 1)
                #print(centroid)
                neuron_properties = np.concatenate((neuron_properties, centroid), 
                                                   axis = 1) if len(neuron_properties) > 0 else centroid
                if not os.path.exists(path4results + 'IDneuronsStackReg/' + str(animal) + '/'):
                    os.makedirs(path4results + 'IDneuronsStackReg/' + str(animal) + '/')
                np.save(path4results + 'IDneuronsStackReg/' + str(animal) + '/' +
                        str(idx) + 'bis', IDneuron)
        else:
            for idx_neuron, stats_neuron in enumerate(stats):
                match = 0
                centroid = np.expand_dims(np.array(stats_neuron['med']), axis=1) ###TODO
                neuron_properties = np.concatenate((neuron_properties, centroid), axis = 1)
                image1 = np.zeros((512, 512))  ###TODO could be integers!!!
                image1[stats_neuron['ypix'], stats_neuron['xpix']] = 1
                size_neuron_image1 = np.count_nonzero(image1)
                cross_image = np.zeros((512, 512))

                #necessary range
                #xmin=int(min(0,centroid[1]-100))  ###TODO check it
                #xmax=int(max(512,centroid[1]+100))
                #ymin=int(min(0,centroid[0]-100))
                #ymax=int(max(512,centroid[0]+100))

                for image_idx in range(idx):
                    image1 = sr.transform(image1, tmat=tmats_loaded[image_idx, idx, :, :])
                    ###image1[image1 > 0] = 1
                    image1[image1 > 0.2] = 1
                    image1[image1 < 0.2] = 0
                    cross_image[:,:]= image[:, :, image_idx] * image1
                    #plt.imshow(cross_image, interpolation='None', cmap='viridis')
                    #plt.show()
                    #sys.exit()
                    counts, bins = np.histogram(image, np.arange(0, np.max(image) + 2))  #slowest part
                    
                
                    counts, bins = np.histogram(image, np.arange(0, np.max(image) + 2))  #slow part
                    counts_cross, bins = np.histogram(cross_image, np.arange(0, np.max(image) + 2)) #slowest part

                    #counts = np.zeros(int(np.max(image)+100), dtype=int)
                    #for pixel in image.flatten():
                    #    counts[int(pixel)]+=1

                    #print(counts_new)
                    #print(int(np.max(image)+1))
                    #print(counts)
                    counts_cross, bins = np.histogram(cross_image, np.arange(0, np.max(image) + 2)) #slowest part
                    #print(len(counts_cross))
                    #counts_cross = np.zeros(int(np.max(image)+100), dtype=int)
                    #print(len(counts_cross))
                    #for pixel in cross_image.flatten():
                    #    print(pixel)
                    #    print(int(pixel))
                    #    counts_cross[int(pixel)]+=1

                    #print(counts)
                    #plt.show()
                    #sys.exit()
                    ratio = counts_cross / counts  ###Is this a problem???
                    #print(ratio)
                    ratio = ratio[1:]
                    ratio_bis = max(counts_cross[1:]) / size_neuron_image1
                    max_ratio = max(np.hstack((ratio, ratio_bis)))
                    # make an empty image with only one dot where centroid is
                    image_centroid = np.zeros((512, 512))
                    image_centroid[int(stats_neuron['med'][0]), 
                                   int(stats_neuron['med'][1])] = 1
                    # transform the image according to registration
                    image_centroid = sr.transform(image_centroid, tmat=tmats_loaded[image_idx, idx, :, :])
                    # find the centroid
                    centroid = np.argmax(image_centroid)
                    centroid = np.unravel_index(centroid, np.shape(image_centroid))
                    # compute euclidean distance between registered centroid and centroid
                    # of neuron whose pixel overlap is maximum (in micrometers)
                    distance = np.max(euclidean_distances((centroid,
                                                           neuron_properties[:, np.argmax(ratio)]))) * 0.41
                    # if both conditions are satisfied, assign same label as neuron that is already present
                   # print(max_ratio,  distance)
                    rat.append(max_ratio)
                    dist.append(distance)

                    if max_ratio > threshold and distance < distance_threshold:
                        IDneuron[idx_neuron] = np.argmax(ratio) + 1
                        image[stats_neuron['ypix'], stats_neuron['xpix'], idx] = IDneuron[idx_neuron]
                        match = 1
                        #print("Match: ", IDneuron[idx_neuron])
                        image_match_neu[stats_neuron['ypix'], stats_neuron['xpix'], idx] = 255 #IDneuron[idx_neuron]
                        #print("match")
                        break # stop searching for neuron
                if match < 1: # if there is no good match, assign a new neuron ID
                    IDneuron[idx_neuron] = neuron_idx
                    image[stats_neuron['ypix'], stats_neuron['xpix'], idx] =  neuron_idx
                    neuron_idx = neuron_idx + 1
            np.save(path4results + 'IDneuronsStackReg/' + str(animal) + '/' +
                            str(idx) + 'bis', IDneuron)

            #file_title = "_" + str(int(idx)) + "_ROI_match"
            #plt.imshow(sr.transform(image_match_neu[:, :, idx], tmat=tmats_loaded[image_idx, idx, :, :]))
            #plt.savefig(save_plots_path + "StackRegVisualInspection/" + file_title + ".png")
            print("saved")
            print("saved", file=f2)
print("number of the unique indexed neurons is ", neuron_idx)
print("number of the unique indexed neurons is ", neuron_idx, file=f2)

f2.close()


    
    