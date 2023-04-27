#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec  4 10:47:20 2021

@author: watsonlevens
"""
from networkx.utils import not_implemented_for
from networkx.utils import py_random_state
import pickle 
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl # For colouring
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
from matplotlib.patches import ConnectionPatch


## loglogPDF p = 0 data
##Numerical results. Opening the degrees
########################################################################################################################
openfile = open('all_all_deg_g1.pkl', 'rb') # N = 1000000 for 5 runs
degrees_g = pickle.load(openfile)          # p = 0, q = 1
#########################################################################################################################

#Theoretical Results
#########################################################################################################################
openfile = open('Proportions_g1.pkl', 'rb') # p = 0, q = 1
Proportions_g = pickle.load(openfile) # Size of proportions = np.max(degrees_g)
#########################################################################################################################

#Log bins . We want 1,2,3 because special case. Then we have 4-6, 7-10
mybins_g= np.unique(np.logspace(np.log10(1),np.log10(np.max(degrees_g)+1), num=30, endpoint=True, base=10.0, dtype=int))
degrees_g=np.ndarray.flatten(np.array(degrees_g))  # To make a flat list(array) out of a list(array) of lists(arrays)
hist_g = np.histogram(degrees_g, bins=mybins_g)
pdf_g =hist_g[0]/np.sum(hist_g[0])#normalize histogram --for pdf

box_sizes=mybins_g[1:]-mybins_g[:-1]
pdf_g = pdf_g/box_sizes # Divide pdf over boxes sizes

#bins = Midpoints of distribution
mid_points_g=np.power(10, np.log10(mybins_g[:-1]) + (np.log10(mybins_g[1:]-1)-np.log10(mybins_g[:-1]))/2)
#mid_points_g=mid_points_g. astype(int)  # Mid_points_g as integers


#Linear binning
ncounts, bins = np.histogram(degrees_g,bins=np.arange(1,np.max(degrees_g),1))
#ncounts, bins = np.histogram(degrees_g,bins=mybins_g)
pdf_g1=ncounts/np.sum(ncounts)  #normalize histogram --for pdf -probability density function       



###EXPONENTS VS CLUSTERING DATA
## Maximum Likelihood(MLE) alpha results
########################################################################################################################
openfile = open('alpha_exponent_MLE.pkl', 'rb') #
alpha_MLE = pickle.load(openfile)         
#####################################
########################################################################################################################
openfile = open('pval_MLE.pkl', 'rb')
pval_MLE = pickle.load(openfile)          
#####################################
########################################################################################################################
openfile = open('qval_MLE.pkl', 'rb') 
qval_MLE = pickle.load(openfile)          
#####################################
##Numerical results. Opening the degrees
########################################################################################################################
openfile = open('Numericalclusters_ML.pkl', 'rb') 
Numericalclusters = pickle.load(openfile)          
######################################



##Plotting the data
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(16, 6),sharex=False, sharey=False, squeeze= True)
                          # If sharex=True and sharey=True, the x and y axis will be shared
plt.rcParams['font.size'] = '22'

ax1.scatter(bins[:-1],pdf_g1,color='green', label = 'Numerical calculations', alpha=0.5) 
ax1.loglog(mid_points_g,pdf_g,color= 'blue',label ='Numerical calculations', linewidth=1, alpha=1)
k_theory = np.arange(0,np.max(degrees_g),1)
ax1.loglog(k_theory,Proportions_g,color='red',label = 'Theoretical calculations',linewidth=1, alpha=1)
ax1.set_xlabel('Degree $k$')
ax1.set_ylabel('Frequency')
ax1.set_xlim([1,np.max(degrees_g)+84522])
ax1.set_ylim([0.0000000000001,1])
##Label figure A
ax1.text(0.2, 3, 'A', fontsize=20,fontweight='bold')

# im = image, # create image: A scatter plot image, 
im = ax2.scatter(Numericalclusters, alpha_MLE, c=pval_MLE*qval_MLE,cmap=mpl.cm.Reds,s=100, alpha=1)
# im = image
# Add a colorbar
colorbar= plt.colorbar(im,ax=ax2, orientation = 'vertical',shrink =1, label ='$C_T$')
# set the colorbar limits - not necessary here, but good to know how.
im.set_clim(0.0, 0.5)
# set ticks - not necessary here, but good to know how.
#colorbar.set_ticks([0.0,0.1,0.2,0.3,0.4,0.47])



ax2.set_xlabel('$C_T$')
ax2.set_ylabel('Power law exponent ($ \u03B1 $)',fontsize=20)
ax2.set_xlim([0,0.6])
ax2.set_ylim([0, 10])
##Label figure B
ax2.text(-0.05, 10.5, 'B', fontsize=20,fontweight='bold')



## Drawing the line with arrows on figure B
prop = dict(arrowstyle="-|>,head_width=0.1,head_length=0.4",
            shrinkA=0,shrinkB=0,facecolor='green',lw=2,ec="blue")
# line for q=1
ax2.annotate("", xy=(0.47,2.8), xytext=(0.0,1.2), arrowprops=prop) 
# the line start at xytext and ends at xy
# line for p=1
ax2.annotate("", xy=(0.15,9.5), xytext=(0.48,4.05), arrowprops=prop) 

##Writing text on top of line p=1 in figure B
ax2.text(0.9,0.4, '$p=1$, $q$ decreasing',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax2.transAxes, rotation=-45)
##Writing text on top of line q=1 in figure B
ax2.text(0.8,0.04, '$q=1$, $p$ increasing',
        verticalalignment='bottom', horizontalalignment='right',
        transform=ax2.transAxes,rotation=14)

#plt.savefig('Model3_Expo_Vs_Clu_combined_LoglogplotPDF_p_zero.pdf')





