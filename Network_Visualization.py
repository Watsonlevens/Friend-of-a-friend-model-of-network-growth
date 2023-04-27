"""
Created on Thu Mar 25 11:17:13 2021

@author: watsonlevens
"""
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np

## DRAWING AND DISPLAYING NETWORKS FOR DIFFERENT RUNS
m0 = 3# Initial number of edges
p = 0 # Probability of attaching to target vertex
q = 1  # Probability of attaching to neighbor vertex
N = 300  # Network size
for n in np.arange (6): #Number of runs or networks to be generated
    Gi =nx.path_graph(m0,create_using=nx.Graph()) # Initial graph with m0 nodes-Return the Path graph P_n of n nodes linearly connected by n-1 edges.
    Gi.add_edge(2, 0) # Add an edge to make a triangle of three connected nodes
   
    
    
    
    for source in range(m0, N): # Start connection from m0 node and stop at N
        node = random.choice(list(Gi.nodes())) # Choose node randomly among the many nodes availabe
        Gi.add_node(source) # Add node to the network
        if p>random.random():   
           Gi.add_edge(source, node) # Add edge between new incoming node and a target node
        if q >random.random():
           nbrs =[nbr for nbr in Gi.neighbors(node)
                            if not Gi.has_edge(source, nbr) # If no edge btn source node and nbr node
                            and not nbr == source] # If neighbor node is not equal to source node 
           
           if len(nbrs)!=0: #Check if neighbor exists then choose one and add edge
               nbr= random.choice(nbrs)
               Gi.add_edge(source, nbr) # Add edge  between the neighbour and the new incoming node                  
#    #Display the graph
#    nx.draw_networkx(Gi, node_size=4, node_color='red', pos=nx.spring_layout(Gi),with_labels = False)
#    #plt.savefig('Network_Display_N' + str(n) + '.pdf')   # Saving figure  
#    plt.show()   

#    #Display the graph
#    nx.draw(Gi, node_size=4, node_color='red', pos=nx.spring_layout(Gi),with_labels = False)
#    #plt.savefig('Network_Display_N' + str(n) + '.pdf')   # Saving figure  
#    #plt.savefig('Model3_p1_p1_Visualization' + str(n) + '.pdf')   # Saving figure  
#    plt.show() 
               
        #Degree centrality visualization       
    pos = nx.spring_layout(Gi)
    degCent = nx.degree_centrality(Gi)
#    eigenCent =nx.eigenvector_centrality(Gi)
    #betCent = nx.betweenness_centrality(Gi, normalized=True, endpoints=True)
    node_color = [20000.0 * Gi.degree(v) for v in Gi]
    node_size =  [v * 10000 for v in degCent.values()]
    plt.figure(figsize=(10,10))
    nx.draw_networkx(Gi, pos=pos, with_labels=False,
                     node_color=node_color,
                     node_size=node_size)
    plt.axis('off')