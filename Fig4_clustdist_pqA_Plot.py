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

# p_Numerical results. Opening the clusters
#########################################################################################################################
openfile = open('p_NumeriClustering_a.pkl', 'rb') 
p_NumeriClustering_a = pickle.load(openfile)
#########################################################################################################################
#Theoretical results
#########################################################################################################################
openfile = open('p_Theorclusters_a.pkl', 'rb') 
p_Theorclusters_a = pickle.load(openfile)
#########################################################################################################################

# q_Numerical results. Opening the clusters
#########################################################################################################################
openfile = open('q_NumeriClustering_a.pkl', 'rb') 
q_NumeriClustering_a = pickle.load(openfile)
#########################################################################################################################
##########################################################################################################################
#q_Theoretical results
#########################################################################################################################
openfile = open('q_Theorclusters_a.pkl', 'rb') 
q_Theorclusters_a = pickle.load(openfile)
#########################################################################################################################

#Random probabilities
#############################################################################################################
##prob =[random.random() for p in range(100)]
openfile = open('probs.pkl', 'rb') 
prob = pickle.load(openfile)


#Plot
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(18, 8), sharey=True, squeeze= True)
# Set general font size
plt.rcParams['font.size'] = '20'
 #Plot
 
 #p-clustering plots
ax1.scatter(prob,p_NumeriClustering_a,color='blue',label = 'Numerical simulation', alpha=0.5,marker="s", s = 25)
ax1.plot(np.sort(prob),np.sort(p_Theorclusters_a),color='red',label = 'Theoretical calculations', alpha=0.5)

ax2.scatter(prob,q_NumeriClustering_a,color='blue',label = 'Numerical simulation', alpha=0.5,marker="s", s = 25)
ax2.plot(np.sort(prob),np.sort(q_Theorclusters_a),color='red',label = 'Theoretical calculations', alpha=0.5)
    
ax1.set(xlabel='Probability $q$', ylabel='$C_T$')
ax2.set(xlabel='Probability $p$', ylabel='$C_T$')


    ## p_Label of panels
ax1.text(-0.21, 0.5, 'A', fontsize=20,fontweight='bold')
ax2.text(-0.18, 0.5, 'B', fontsize=20,fontweight='bold')
#plt.savefig('pqA_cluring_distribution_model3.pdf')
