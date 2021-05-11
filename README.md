# Anesthetics uniquely decorrelate hippocampal network activity, alter spine dynamics and affect memory consolidation

This repository contains code related to the paper "Anesthetics fragment hippocampal network activity, alter spine dynamics and affect memory consolidation".
The manuscript is available here: https://www.biorxiv.org/content/10.1101/2020.06.05.135905v1

## Analysis pipeline

Click on the image below to open it in interactive mode: 

[<img src="Yang_Chini_software.svg">](https://mchini.github.io/Yang_Chini_et_al/Yang_Chini_software.html)

### Alignment of recordings across imaging sessions

Code to align calcium imaging recordings from different sessions but with the same FOV is in the [Alignment](https://github.com/mchini/Yang_Chini_et_al/tree/master/Alignment%20Scripts%20(Python)) folder.

Before alignment             |  After alignment
:-------------------------:|:-------------------------:
![](no_alignment.gif)  |  ![](with_alignment.gif)


### Analysis of calcium transients and their correlations

Python code for analysis of calcium transients and correlation matrices is in [Figures 3-5](https://github.com/mchini/Yang_Chini_et_al/tree/master/Figures%203-5%20(Python)) folder

![](correlations_small.png)

### Clustering and classification

Matlab code for clustering in the spatial and temporal domain and sleep classification is in the [Figures 5-7](https://github.com/mchini/Yang_Chini_et_al/tree/master/Figures%205-7%20(MATLAB)) folder

![](clustering.png)

### Electrophysiology

Further Matlab code that was used for the ephys-part of the paper can be found in this other [repository](https://github.com/mchini/HanganuOpatzToolbox)

![](ephys_small.png)

R scripts and datasets that were used for all statistical analysis are available in the [Stats](https://github.com/mchini/Yang_Chini_et_al/tree/master/Stats%20(R)) folder.

Raw 2-photon and electrophysiology data is available at this [repository](https://gin.g-node.org/SW_lab/Anesthesia_CA1/) on GIN.

## Notebooks

**Validation of calcium recordings: homogeneity and stability**

This notebook illustrates the extraction of the main features of the calcium traces for a large set of recordings and subsiquest analysis it's variations and stability.    

(notebooks/Validation_stability.ipynb)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mchini/Yang_Chini_et_al/blob/testTransition/notebooks/Validation_stability.ipynb)

**Validation of calcium recordings field of view alignment)**

This notebook contains the code for the aligment of field of view between recordings.

(notebooks/Validation_FOV_alignment.ipynb)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mchini/Yang_Chini_et_al/blob/testTransition/notebooks/Validation_FOV_alignment.ipynb)

**Validation of calcium recordings: motion)**

(notebooks/Validation_motion.ipynb)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mchini/Yang_Chini_et_al/blob/testTransition/notebooks/Validation_motion.ipynb)

**Analysis of neuronal activity on single neuron level**

Extraction of the main features permits the analysis of the recordings on both population and single neuron levels.
 
(notebooks/single-neuron-level_analysis.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mchini/Yang_Chini_et_al/blob/testTransition/notebooks/single-neuron-level_analysis.ipynb)


