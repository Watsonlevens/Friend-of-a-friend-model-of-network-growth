"""
Created on Thu Mar 25 11:17:13 2021

@author: watsonlevens
"""
import pickle 
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np

m0 = 3# Initial number of edges
p = 0 # Probability of attaching to target vertex
q = 1  # Probability of attaching to neighbor vertex
N = 1000000 # Network size
G =nx.path_graph(m0,create_using=nx.Graph()) # Initial graph with m nodes-Return the Path graph P_n of n nodes linearly connected by n-1 edges.
# A function for generating graph
def graph(Gi,N,m0):   #Gi---Initial graph
    Gi.add_edge(2, 0)
    for source in range(m0, N): # Start connection from m0 node and stop at N
        node = random.choice(list(Gi.nodes())) # Choose node randomly among the many nodes availabe
        Gi.add_node(source) # Add node to the network
        if p >random.random():      
           Gi.add_edge(source, node) # Add edge 
        if q >random.random():
           nbrs =[nbr for nbr in Gi.neighbors(node)
                            if not Gi.has_edge(source, nbr) # If no edge btn source node and nbr node
                            and not nbr == source] # If neighbor node is not equal to source node 
           if len(nbrs)!=0: #Check if neighbor exists then choose one and add edge
               nbr= random.choice(nbrs)
               Gi.add_edge(source, nbr) # Add edge   
    return Gi
## A function to calculate degrees of nodes
#######################################################################################################################
def get_all_degree(G):
    all_degrees= G.degree() #All the in_degrees
    all_deg= np.array(list(nod_degres for nod, nod_degres in all_degrees)) 
    return all_deg

# Making a list of degree values for several graphs and draw the graph that take the average of all degrees. 
all_all_deg=[]  
for rep in range(5): # Reproduce the graph several times and draw the mean of data 
    G =nx.path_graph(m0,create_using=nx.Graph()) 
    Network = graph(G,N,m0) 
    all_deg=get_all_degree(Network)
    all_all_deg.extend(all_deg) 
    
## Saving all_all_deg    
#fileName   =  'all_all_deg' + ".pkl";
#file_pi = open(fileName, 'wb') 
#pickle.dump(all_all_deg, file_pi)