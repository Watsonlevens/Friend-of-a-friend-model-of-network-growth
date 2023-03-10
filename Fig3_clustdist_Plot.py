#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 21:25:56 2021

@author: watsonlevens
"""
import pickle 
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
from networkx.algorithms.approximation import average_clustering
from networkx.utils import not_implemented_for
from networkx.utils import py_random_state
##########################################################################################################
#CLUSTERING AS A FUNCTION OF DEGREE
#############################################################################################################

# q_Numerical results. Opening the clusters
#########################################################################################################################
openfile = open('k_NumeriClustering_a.pkl', 'rb') 
k_NumeriClustering_a = pickle.load(openfile)
#########################################################################################################################
#########################################################################################################################
openfile = open('k_NumeriClustering_b.pkl', 'rb') 
k_NumeriClustering_b = pickle.load(openfile)
#########################################################################################################################

#########################################################################################################################
openfile = open('k_NumeriClustering_c.pkl', 'rb') 
k_NumeriClustering_c = pickle.load(openfile)
#########################################################################################################################

#########################################################################################################################
openfile = open('k_NumeriClustering_d.pkl', 'rb') 
k_NumeriClustering_d = pickle.load(openfile)
#########################################################################################################################

#########################################################################################################################
openfile = open('k_NumeriClustering_e.pkl', 'rb') 
k_NumeriClustering_e = pickle.load(openfile)
#########################################################################################################################

#########################################################################################################################
openfile = open('k_NumeriClustering_f.pkl', 'rb') 
k_NumeriClustering_f = pickle.load(openfile)
#########################################################################################################################



#q_Theoretical results
#########################################################################################################################
openfile = open('k_Theorclusters_a.pkl', 'rb') 
k_Theorclusters_a = pickle.load(openfile)
#########################################################################################################################

#########################################################################################################################
openfile = open('k_Theorclusters_b.pkl', 'rb') 
k_Theorclusters_b = pickle.load(openfile)
#########################################################################################################################
openfile = open('k_Theorclusters_c.pkl', 'rb') 
k_Theorclusters_c = pickle.load(openfile)

openfile = open('k_Theorclusters_d.pkl', 'rb') 
k_Theorclusters_d = pickle.load(openfile)
##########################################################################################################

openfile = open('k_Theorclusters_e.pkl', 'rb') 
k_Theorclusters_e = pickle.load(openfile)

openfile = open('k_Theorclusters_f.pkl', 'rb') 
k_Theorclusters_f = pickle.load(openfile)
##########################################################################################################



#Setting minimum degree to have x and y axis with equal data points in the plots
xlimit = min(len(k_NumeriClustering_a),len(k_NumeriClustering_b),len(k_NumeriClustering_c),
       len(k_NumeriClustering_d),len(k_NumeriClustering_e),len(k_NumeriClustering_f))
k =np.arange(xlimit)


#fig,ax1=plt.subplots(1,1)
#fig,(ax1,ax2)=plt.subplots(2)
#fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2,figsize=(16, 18), sharex=True, sharey=True, squeeze= True)
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2,figsize=(16, 18), sharex=True, sharey=True, squeeze= True)
#ax.set_xlabel('Degree $k$')
#ax.set_ylabel('Frequency')
# Set general font size
plt.rcParams['font.size'] = '20'

 ## CLUSTERING IS DEFINED FOR k>=3, SO WE PLOT THE DATA FROM k=3
ax1.scatter(k[3:],k_NumeriClustering_a[3:xlimit],color='blue',label = 'Numerical simulation', alpha=0.5,marker="s", s = 25)
ax1.plot(k[3:],k_Theorclusters_a[3:xlimit],color='red',label = 'Theoretical calculations', alpha=0.5)

ax2.scatter(k[3:],k_NumeriClustering_b[3:xlimit],color='blue',label = 'Numerical simulation', alpha=0.5,marker="s", s = 25)
ax2.plot(k[3:],k_Theorclusters_b[3:xlimit],color='red',label = 'Theoretical calculations', alpha=0.5)
#
ax3.scatter(k[3:],k_NumeriClustering_c[3:xlimit],color='blue',label = 'Numerical simulation', alpha=0.5,marker="s", s = 25)
ax3.plot(k[3:],k_Theorclusters_c[3:xlimit],color='red',label = 'Theoretical calculations', alpha=0.5)

ax4.scatter(k[3:],k_NumeriClustering_d[3:xlimit],color='blue',label = 'Numerical simulation', alpha=0.5,marker="s", s = 25)
ax4.plot(k[3:],k_Theorclusters_d[3:xlimit],color='red',label = 'Theoretical calculations', alpha=0.5)

ax5.scatter(k[3:],k_NumeriClustering_e[3:xlimit],color='blue',label = 'Numerical simulation', alpha=0.5,marker="s", s = 25)
ax5.plot(k[3:],k_Theorclusters_e[3:xlimit],color='red',label = 'Theoretical calculations', alpha=0.5)

ax6.scatter(k[3:],k_NumeriClustering_f[3:xlimit],color='blue',label = 'Numerical simulation', alpha=0.5,marker="s", s = 25)
ax6.plot(k[3:],k_Theorclusters_f[3:xlimit],color='red',label = 'Theoretical calculations', alpha=0.5)





for ax in fig.get_axes():
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	    label.set_fontsize(20)
    ax.set(xlabel='Degree $k$', ylabel='$C_k$')
    #ax.set_xscale('log')
    #ax.set_yscale('log')
    #ax.legend(loc="upper right")
    ax.label_outer()  # Set axis scales outer
    #ax.get_xlim()[1:200]
    #ax.get_ylim()[0.000001:1]
    
# Set common labels
#fig.text(0.5, 0.04, 'Degree $k$', ha='center', va='center')
#fig.text(0.06, 0.5, 'Frequency', ha='center', va='center', rotation='vertical')
#
#    ## k_Label of panels
ax1.text(0, 0.7, 'A', fontsize=20,fontweight='bold')
ax2.text(0.5, 0.7, 'B', fontsize=20,fontweight='bold')
ax3.text(0, 0.7, 'C', fontsize=20,fontweight='bold')
ax4.text(0.5, 0.7, 'D', fontsize=20,fontweight='bold')
ax5.text(0, 0.7, 'E', fontsize=20,fontweight='bold')
ax6.text(0.5, 0.7, 'F', fontsize=20,fontweight='bold')


#plt.xlim([1,200])  # Limiting x axis
#plt.ylim([0,1])  # Limiting x axis

#plt.savefig('k_cluring_distribution_model3.pdf')
