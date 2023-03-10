#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 22:45:15 2021

@author: watsonlevens
"""
import numpy as np
import pickle 
import networkx as nx
import random
import math
import scipy
from scipy import special
import scipy.special as sc
#from scipy.stats import gamma
import matplotlib.pyplot as plt
# CLUSTERING AS A FUNCTION OF degree k          
p = 1
q = 1

# Theoretical calculations 
k=np.arange(3,10000)   ## CLUSTERING IS DEFINED FOR k>=3, SO WE WILL CONSIDER PLOT OF DATA FROM k=3

c_p = (4*(p+q)+(2*k))/((2*k*(p+q)/q)+(k*k/p)) # Clustering as function of p,q and k

### Saving the clusterings        
#######################################################################################################################
#TheorfileName   =  'k_Theorclusters_a' + ".pkl";
#Theofile_pi = open(TheorfileName, 'wb') 
#pickle.dump(c_p, Theofile_pi)
#######################################################################################################################
#



#Ploting clustering
###############################################################################
plt.plot(k,c_p,color='red',label = 'Theoretical calculations', alpha=0.5)
#################################################################################
























