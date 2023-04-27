#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 25 10:53:09 2021

@author: watsonlevens
"""
import pickle 
import networkx as nx
import random
import math
import scipy
from scipy import special
import scipy.special as sc
#from scipy.stats import gamma
import matplotlib.pyplot as plt
import numpy as np
p = 0 # Probability of attaching to target vertex
q = 1  # Probability of attaching to neighbor vertex
# Computing proportions of nodes with degree k

#Master equation approachformula approximation ----1
k=np.arange(0,915478,1) # Degrees  95743
P=np.zeros(len(k))
EX=2*(p+q) # Expected degree to add each time
P[0]=((1-q)*(1-p))/(1+p) # Proportion of nodes with degree k = 0
P[1]=(p*(1-q)+q*(1-p)+p*P[0])/(1+p+q/EX) # Proportion of nodes with degree k = 1
P[2]=(p*q+(p+q/EX)*P[1])/(1+p+q*2/EX) # Proportion of nodes with degree k = 2
for kc in k[3:]: # Proportion of nodes with degree k>=3
    P[kc]= (p+q*(kc-1)/EX)/(1+p+q*kc/EX)*P[kc-1]

#
# Saving the P
######################################################################################################################
fileName   =  'Proportions_g1' + ".pkl";
file_pi = open(fileName, 'wb') 
pickle.dump(P, file_pi)
########################################################################################################################
########################################################################################################################
fig, ax = plt.subplots(1, figsize=(10, 6), sharex=True, sharey=True, squeeze= True)

#Plot master equation approach formula---1
ax.loglog(k,P, alpha=0.5,label = 'Master equation approximation 1', color='red')
plt.legend(loc="lower left")
ax.set(xlabel= "$k$", ylabel='Frequency')
ax.set_xlim(1, 1000000)
ax.set_ylim(0, 1)