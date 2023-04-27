
#from networkx.algorithms.approximation import average_clustering
from networkx.utils import not_implemented_for
from networkx.utils import py_random_state
import pickle 
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np

## MODEL3-CHOOSE NODE RANDOMLY AND CONNECT TO BOTH THE NODE AND NEIGHBOR RANDOMLY



def graph(Gi,N,m0):   #Gi---Initial graph
                Gi.add_edge(2, 0)
                #Gi.add_node(source) # Add node
                #display_gaph_stepwise(Gi)
                for source in range(m0, N): # Start connection from m0 node and stop at n
                    node = random.choice(list(Gi.nodes())) # Choose node randomly among the many nodes availabe
                    Gi.add_node(source) # Add add to the network
                    if p>random.random():
                        
                       Gi.add_edge(source, node)
                       #Gi.add_edge(node, source)
                    #else:
                    if q >random.random():
                       nbrs =[nbr for nbr in Gi.neighbors(node) # neighborhoods are nodes followed by target
                                        if not Gi.has_edge(source, nbr) # If no edge btn source node and nbr node (followed node)
                                        and not nbr == source] # If neighbor node is not equal to source node 
                       if len(nbrs)!=0:
                           nbr= random.choice(nbrs)
                           #Gi.add_edge(nbr,source)
                           Gi.add_edge(source, nbr) # Add edge   
            #        if not Gi.has_edge(source, node):
            #            print(node)
                    #display_gaph_stepwise(Gi)   
                return Gi
            
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
        #print('iteration no:',i)
        #Can take same node twice or more. Can be a problem on small graphs.
        #FOLLOWERS AND FOLLOWED, Weakly connected
        nbrs = list(G[nodes[i]])
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


##prob =[random.random() for p in range(100)]
#openfile = open('probs.pkl', 'rb') 
#prob = pickle.load(openfile)
prob =[random.random() for p in range(100)]
N = 100# Network  size
m0 = 3# Initial number of edges
p = 1
p_NumeriClustering =[] # Numerical
for q in prob:            
    Gi =nx.path_graph(m0,create_using=nx.Graph()) # Initial graph with m nodes-Return the Path graph P_n of n nodes linearly connected by n-1 edges.
    FOF = graph(Gi,N,m0)
    #Numerical calculations of clustering
    ################################################################################ 
    
    node_clustering(FOF, trials=1000, seed=None)
    numtrials=1000
    Degrees_and_clustering_coefficients =node_clustering(graph(Gi,N,m0),trials=1000)
    array_Degrees_and_clustering_coefficients=np.array(Degrees_and_clustering_coefficients)
    clustering_coefficients=array_Degrees_and_clustering_coefficients[:,1]
    degree=array_Degrees_and_clustering_coefficients[:,0]
    
    
    #Find nodes with degree k and calculate their mean
    max_k=max(array_Degrees_and_clustering_coefficients[:,0])
    
    
    meancluster=np.zeros(int(max_k))  # numpy mean clusters
    propcluster=np.zeros(int(max_k)) # numpy proportional of clusters            
    for k_i in np.arange(max_k):
        
        #Find nodes with degree k
        withdegreek_i=array_Degrees_and_clustering_coefficients[:,0]==k_i
        #print(withdegreek_i)
        # Computer mean cluster of nodes with degree k
        meancluster[int(k_i)]=np.mean(array_Degrees_and_clustering_coefficients[withdegreek_i,1])
        # Compute proportional of clusters for nodes with degree k.
        propcluster[int(k_i)]=np.sum(withdegreek_i)/numtrials

    #print('Sum of proportionals',sum(propcluster))
    ## Replacing NaN with 0
    meancluster=np.nan_to_num(meancluster)

    #Frequency of numerical clusters with degree k.
    #print('Sum of numerproportional:',np.sum(propcluster)) 
# Normalize with sum of prportional of nodes with k>=3 (i.e sum(propcluster[3:]))
    clustering=sum(meancluster[3:]*propcluster[3:])/sum(propcluster[3:])
    #print('Global clustering',clustering)
    p_NumeriClustering.append(clustering)
  
                   
### Saving the clusterings
########################################################################################################################
#fileName   =  'p_NumeriClustering_d' + ".pkl";
#file_pi = open(fileName, 'wb') 
#pickle.dump(NumeriClustering, file_pi)
########################################################################################################################
     
        
     #Sort according to the size of p or q
indices=np.argsort(prob)

prob=np.array(prob)
prob=prob[indices]

p_NumeriClustering=np.array(p_NumeriClustering)

p_NumeriClustering=p_NumeriClustering[indices]       
#Ploting clustering
plt.scatter(prob,p_NumeriClustering, marker = "+", color = 'red',label = 'Numerical simulation', s= 10)
##plt.xscale('log')
#plt.yscale('log')
#plt.grid(False)

plt.ylabel("$C(q)$", fontsize=12)  
plt.xlabel("Probability $q$", fontsize=12) 
#
#plt.legend(loc="lower right")
#plt.xlim([0,1])  # Limiting x axis
#plt.ylim([0,1])  # Limiting x axis
#plt.savefig('q_Clustering_model3.pdf')
























