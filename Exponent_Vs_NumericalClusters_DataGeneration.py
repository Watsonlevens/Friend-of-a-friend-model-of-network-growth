"""
Created on Tue Aug  3 22:23:05 2021

@author: watsonlevens
"""
from networkx.utils import not_implemented_for
from networkx.utils import py_random_state
import networkx as nx
import random
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl # For colouring

# Numerical calculations of clustering and the power law exponent
prob=np.arange(0,1.01,0.01) # probabilities of linking to a target node and its neighbour
N =100  # Network size
m0 = 3  # Initial network size
 ##A function for generating friend of a friend network
def graph(Gi,N,m0):   #Gi---Initial graph
    Gi.add_edge(2, 0)  # Complete triangle
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

# A function for calculating clustering
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
        if degree > 1: #Calculate clustering for nodes with degree greater than 2
           #print(triangles)
           degree_and_triangles.append([degree,(triangles/(degree*(degree-1)/2))])  
        else:
             degree_and_triangles.append([degree,0])
    return degree_and_triangles 


alpha_exponent_Numerical =np.zeros((len(prob),len(prob)))
Numericalclusters = np.zeros((len(prob),len(prob)))

pval = np.zeros((len(prob),len(prob))) #For coloring
qval = np.zeros((len(prob),len(prob)))#For coloring
kmaxs = np.zeros(len(prob)) # max degrees for each graph to be used in kmax in theoretical
count =0
for u, p in enumerate(prob):
    count +=1
    #print('Count:',count)
    for v,q in enumerate(prob):
        G =nx.path_graph(m0,create_using=nx.Graph()) # Initial graph with m nodes-Return the Path graph P_n of n nodes linearly connected by n-1 edges.    
        Network =  graph(G,N,m0)
        
        ### Computing numerical power law exponents
        all_degrees= Network.degree() #All degrees
        all_deg= np.array(list(nod_degres for nod, nod_degres in all_degrees))
        x_min =10
        kmaxs[u] =np.max(all_deg)
        some_deg = all_deg[all_deg>=x_min]    ##remove zeros from np.array 
        n = len(np.array(some_deg)) 
        if n>4:
            alpha = 1 + n/sum(np.log((some_deg)/ (x_min-0.5))) ##calculate alpha
            alpha_exponent_Numerical[u,v] = alpha
        else:
            continue
        
##        #For coloring
        pval[u,v]=p
        qval[u,v]=q
                #Numerical calculations of clustering
        ################################################################################ 
        numtrials =1000
        Degrees_and_clustering_coefficients =node_clustering(Network, numtrials)
        array_Degrees_and_clustering_coefficients=np.array(Degrees_and_clustering_coefficients)
        clustering_coefficients=array_Degrees_and_clustering_coefficients[:,1]
        degree=array_Degrees_and_clustering_coefficients[:,0]

        #Find nodes with degree k and calculate their mean
        max_k=max(array_Degrees_and_clustering_coefficients[:,0])        
        if max_k > 3:        
            meancluster = np.zeros(int(max_k))        
            propcluster = np.zeros(int(max_k)) # Proportional of clusters            
            for k_i in np.arange(max_k):
                
                #Find nodes with degree k
                withdegreek_i=array_Degrees_and_clustering_coefficients[:,0]==k_i

                # Computer mean cluster of nodes with degree k
                meancluster[int(k_i)]=np.mean(array_Degrees_and_clustering_coefficients[withdegreek_i,1])
                # Compute proportional of clusters for nodes with degree k.
                propcluster[int(k_i)]=np.sum(withdegreek_i)/numtrials
            
  
            meancluster=np.nan_to_num(meancluster) ## Replacing NaN with 0             
            #Frequency of numerical clusters with degree k.
            clustering=sum(meancluster[3:]*propcluster[3:])/sum(propcluster[3:])
            clustering = np.nan_to_num(clustering)
            Numericalclusters[u,v]= clustering
        else :
            continue
# Plotting the feasible region
fig, ax = plt.subplots(1,figsize=(6, 6), sharex=True, sharey=True, squeeze= True)
plt.rcParams['font.size'] = '20'
#ax.scatter(Numericalclusters, alpha_exponent_Numerical, s=100, cmap=mpl.cm.Reds, alpha=0.5)
ax.scatter(Numericalclusters, alpha_exponent_Numerical,c=pval*qval, s=100, cmap=mpl.cm.Reds, alpha=0.5)
ax.set(xlabel= "$C_T$", ylabel='Power law exponent ($ \u03B1 $)')
ax.set_xlim(0,1)
ax.set_ylim(0, 10)
#plt.savefig('Exponent_clustering.pdf')

