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
fig,ax2=plt.subplots(1,1,figsize=(7, 6),sharex=False, sharey=False, squeeze= True)
                          # If sharex=True and sharey=True, the x and y axis will be shared
plt.rcParams['font.size'] = '22'
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


###Label figure B
#ax2.text(-0.05, 10.5, 'B', fontsize=20,fontweight='bold')



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

#plt.savefig('Model3_Expo_Vs_Clustering_combined_LoglogplotPDF_p_zero.pdf')





