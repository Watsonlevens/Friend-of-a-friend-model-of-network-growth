#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 22:45:15 2021

@author: watsonlevens
"""



#from networkx.algorithms.approximation import average_clustering
from networkx.utils import not_implemented_for
from networkx.utils import py_random_state
import pickle 
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
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



















##Ploting clustering
################################################################################
#plt.plot(k,c_p,color='red',label = 'Theoretical calculations', alpha=0.5)
##################################################################################
##
#plt.scatter(k,meancluster[3:], marker = "+", color = 'blue',label = 'Numerical simulation', s= 10)
###plt.xscale('log')
##plt.yscale('log')
##plt.grid(False)
#
#plt.ylabel("$C(k)$", fontsize=12)  
#plt.xlabel("Degree $k$", fontsize=12) 
##
##plt.legend(loc="upper right")
##plt.xlim([0,1])  # Limiting x axis
##plt.ylim([0,1])  # Limiting x axis
##plt.savefig('q_Clustering_model3.pdf')
###plt.scatter(HK_interval_centers[2:],HK_average_list[2:], marker='o')
#























