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
N = 1000000# Network  size
m0 = 3# Initial number of edges
p = 1
q = 0.75
Gi =nx.path_graph(m0,create_using=nx.Graph()) # Initial graph with m nodes-Return the Path graph P_n of n nodes linearly connected by n-1 edges.
def graph(Gi,N,m0):   #Gi---Initial graph
    Gi.add_edge(2, 0)
    for source in range(m0, N): # Start connection from m0 node and stop at n        
        node = random.choice(list(Gi.nodes())) # Choose node randomly among the many nodes availabe
        Gi.add_node(source) # Add add to the network
        if p>random.random():
            
           Gi.add_edge(source, node)
        if q >random.random():
           nbrs =[nbr for nbr in Gi.neighbors(node) # neighborhoods are nodes followed by target
                            if not Gi.has_edge(source, nbr) # If no edge btn source node and nbr node (followed node)
                            and not nbr == source] # If neighbor node is not equal to source node 
           if len(nbrs)!=0:
               nbr= random.choice(nbrs)
               Gi.add_edge(source, nbr) # Add edge     
    return Gi


#Numerical calculations of clustering
################################################################################ 
@py_random_state(2)
@not_implemented_for('directed') 
def node_clustering(G, trials=1000, seed=None):
    #Calculates  [trials] weakly connected triangle clusters in graph G
    n = len(G)
    #triangles = 0
    nodes = list(G)
    
    #List of all degree, triangle pairs
    degree_and_triangles=[]
    
    
    for i in [int(seed.random() * n) for i in range(trials)]:
        nbrs = list(G[nodes[i]])
        #print('Neighbors of ',i,'are',list(nbrs))
        degree=len(nbrs)
        
        triangles=0
        
#            
        for j,u in enumerate(nbrs):
            #Find all pairs
            for v in nbrs[j+1:]:
                if (u in G[v]):
                    #Weakly connected
                    triangles += 1
        if degree > 1:
           #print(triangles)
           degree_and_triangles.append([degree,(triangles/(degree*(degree-1)/2))])  
        else:
             degree_and_triangles.append([degree,0])
    return degree_and_triangles 

Degrees_and_clustering_coefficients =node_clustering(graph(Gi,N,m0),trials=1000)
array_Degrees_and_clustering_coefficients=np.array(Degrees_and_clustering_coefficients)
clustering_coefficients=array_Degrees_and_clustering_coefficients[:,1]
degree=array_Degrees_and_clustering_coefficients[:,0]

clustering_coefficients=array_Degrees_and_clustering_coefficients[:,1]

#Find nodes with degree k and calculate their mean
max_k=max(array_Degrees_and_clustering_coefficients[:,0])
# Mean degree-dependent local clustering of nodes with degree k
meancluster=np.zeros(int(max_k))                
for k_i in np.arange(3,max_k):
    
    #Find nodes with degree k
    withdegreek_i=array_Degrees_and_clustering_coefficients[:,0]==k_i
    # Compute mean degree-dependent local clustering of nodes with degree k
    meancluster[int(k_i)]=np.mean(array_Degrees_and_clustering_coefficients[withdegreek_i,1])

## Saving the clusterings
########################################################################################################################
#fileName   =  'k_NumeriClustering_b' + ".pkl";
#file_pi = open(fileName, 'wb') 
#pickle.dump(meancluster, file_pi)
########################################################################################################################
##



















