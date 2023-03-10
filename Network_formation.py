"""
Created on Thu Mar 25 11:17:13 2021
@author: watsonlevens
"""
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np

## DRAWING AND DISPLAYING NETWORKS FOR DIFFERENT REALIZATIONS
m0 = 3# Initial number of edges
p = 0 # Probability of attaching to target vertex
q = 1  # Probability of attaching to neighbour vertex
N = 300  # Network size
Gi =nx.path_graph(m0,create_using=nx.Graph()) # Initial graph with m0 nodes-Return the Path graph P_n of n nodes linearly connected by n-1 edges.
Gi.add_edge(2, 0) # Add an edge to make a triangle of three connected nodes    

for new_node in range(m0, N): # Start connection from m0 node and stop at N
    node = random.choice(list(Gi.nodes())) # Choose node randomly among the many nodes availabe
    Gi.add_node(new_node) # Add node to the network
    if p>random.random():   
       Gi.add_edge(new_node, node) # Add edge between new incoming node and a target node
    if q >random.random():
       nbrs =[nbr for nbr in Gi.neighbors(node)
                        if not Gi.has_edge(new_node, nbr) # If no edge btn source node and nbr node
                        and not nbr == new_node] # If neighbor node is not equal to source node 
       
       if len(nbrs)!=0: #Check if neighbour exists then choose one and add edge
           nbr= random.choice(nbrs)
           Gi.add_edge(new_node, nbr) # Add edge  between the neighbour and the new incoming node                  


#Visualize the graph the graph
nx.draw_networkx(Gi, node_size=4, node_color='red', pos=nx.spring_layout(Gi),with_labels = False)
#plt.savefig('Network_Display_N' + str(n) + '.pdf')   # Saving figure  
plt.show()   

