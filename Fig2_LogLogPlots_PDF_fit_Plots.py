"""
Created on Thu Jul 29 07:55:13 2021

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

# Numerical results. Opening the degrees
#########################################################################################################################
openfile = open('all_all_deg_a.pkl', 'rb') # p=1, q = 1
degrees_a = pickle.load(openfile)
degrees_a=np.ndarray.flatten(np.array(degrees_a))  # To make a flat list(array) out of a list(array) of lists(arrays)

#########################################################################################################################
openfile = open('all_all_deg_b.pkl', 'rb') # p=1, q = 0.75
degrees_b = pickle.load(openfile)
degrees_b=np.ndarray.flatten(np.array(degrees_b))  # To make a flat list(array) out of a list(array) of lists(arrays)

#########################################################################################################################
openfile = open('all_all_deg_c.pkl', 'rb') # p=1, q = 0.25
degrees_c = pickle.load(openfile)
degrees_c=np.ndarray.flatten(np.array(degrees_c))  # To make a flat list(array) out of a list(array) of lists(arrays)

########################################################################################################################
openfile = open('all_all_deg_d.pkl', 'rb') # p=0.75, q = 1
degrees_d = pickle.load(openfile)
degrees_d=np.ndarray.flatten(np.array(degrees_d))  # To make a flat list(array) out of a list(array) of lists(arrays)

#########################################################################################################################
openfile = open('all_all_deg_e.pkl', 'rb') # p=0.25, q = 1
degrees_e = pickle.load(openfile)
degrees_e=np.ndarray.flatten(np.array(degrees_e))  # To make a flat list(array) out of a list(array) of lists(arrays)

#########################################################################################################################
openfile = open('all_all_deg_f.pkl', 'rb') # p=0.5, q = 0.5
degrees_f = pickle.load(openfile)
degrees_f=np.ndarray.flatten(np.array(degrees_f))  # To make a flat list(array) out of a list(array) of lists(arrays)



#########################################################################################################################
openfile = open('all_all_deg_g.pkl', 'rb') # p=0, q = 1
degrees_g = pickle.load(openfile)
degrees_g=np.ndarray.flatten(np.array(degrees_g))  # To make a flat list(array) out of a list(array) of lists(arrays)
#########################################################################################################################
 





#Theoretical Results
#############################################################################################################
#########################################################################################################################
openfile = open('Proportions_a.pkl', 'rb') # p=1, q = 1
Proportions_a = pickle.load(openfile)
#########################################################################################################################
#########################################################################################################################
openfile = open('Proportions_b.pkl', 'rb') # p=1, q = 0.75
Proportions_b = pickle.load(openfile)
#########################################################################################################################

#########################################################################################################################
openfile = open('Proportions_c.pkl', 'rb') # p=1, q = 0.25
Proportions_c = pickle.load(openfile)
#########################################################################################################################
#########################################################################################################################
openfile = open('Proportions_d.pkl', 'rb') # p=0.75, q = 1
Proportions_d = pickle.load(openfile)
#########################################################################################################################


#########################################################################################################################
openfile = open('Proportions_e.pkl', 'rb') # p=0.25, q = 1
Proportions_e = pickle.load(openfile)
#########################################################################################################################
#########################################################################################################################
openfile = open('Proportions_f.pkl', 'rb') # p=0.5, q = 0.5
Proportions_f = pickle.load(openfile)
#########################################################################################################################


#########################################################################################################################
openfile = open('Proportions_g.pkl', 'rb') # p=0, q = 1
Proportions_g = pickle.load(openfile)
#########################################################################################################################




fig, ((ax1, ax2), (ax3, ax4),(ax5, ax6)) = plt.subplots(3,2,figsize=(16, 18), sharex=True, sharey=True, squeeze= True)
plt.rcParams['font.size'] = '20'

##p=1, q =1
##Binning
mybins_a= np.unique(np.logspace(np.log10(1),np.log10(np.max(degrees_a)+1), num=25, endpoint=True, base=10.0, dtype=int))
degrees_a=np.ndarray.flatten(np.array(degrees_a))  # To make a flat list(array) out of a list(array) of lists(arrays)
hist_a = np.histogram(degrees_a, bins=mybins_a)
pdf_a =hist_a[0]/np.sum(hist_a[0])#normalize histogram --for pdf
box_sizes=mybins_a[1:]-mybins_a[:-1]  # size of boxes
pdf_a = pdf_a/box_sizes   # Divide pdf by its box size

#bins = Midpoints of distribution
mid_points_a=np.power(10, np.log10(mybins_a[:-1]) + (np.log10(mybins_a[1:]-1)-np.log10(mybins_a[:-1]))/2)
#mid_points_g=mid_points_g. astype(int)  # Mid_points_g as integers
#Linear binning
#ncounts_a, bins_a = np.histogram(degrees_a,bins=np.arange(1,np.max(degrees_a),1))
#pdf_a1=ncounts_a/np.sum(ncounts_a)  #normalize histogram --for pdf -probability density function       
#ax1.scatter(bins_a[:-1],pdf_a1,color='green', label = 'Numerical calculations', alpha=0.5) 
ax1.loglog(mid_points_a,pdf_a,color= 'blue',label ='Numerical calculations', linewidth=1, alpha=1)
k_theory_a = np.arange(0,np.max(degrees_a),1)
ax1.loglog(k_theory_a,Proportions_a,color='red',label = 'Theoretical calculations',linewidth=1, alpha=1)



#p=1, q =0.75
#Binning
mybins_b= np.unique(np.logspace(np.log10(1),np.log10(np.max(degrees_b)+1), num=30, endpoint=True, base=10.0, dtype=int))
degrees_b=np.ndarray.flatten(np.array(degrees_b))  # To make a flat list(array) out of a list(array) of lists(arrays)
hist_b = np.histogram(degrees_b, bins=mybins_b)
pdf_b =hist_b[0]/np.sum(hist_b[0])#normalize histogram --for pdf
box_sizes=mybins_b[1:]-mybins_b[:-1]  # size of boxes
pdf_b = pdf_b/box_sizes   # Divide pdf by its box size

#bins = Midpoints of distribution
mid_points_b=np.power(10, np.log10(mybins_b[:-1]) + (np.log10(mybins_b[1:]-1)-np.log10(mybins_b[:-1]))/2)
#mid_points_g=mid_points_g. astype(int)  # Mid_points_g as integers
#Linear binning
#ncounts_b, bins_b = np.histogram(degrees_b,bins=np.arange(1,np.max(degrees_b),1))
#pdf_b1=ncounts_b/np.sum(ncounts_b)  #normalize histogram --for pdf -probability density function       
#ax2.scatter(bins_b[:-1],pdf_b1,color='green', label = 'Numerical calculations', alpha=0.5) 
ax2.loglog(mid_points_b,pdf_b,color= 'blue',label ='Numerical calculations', linewidth=1, alpha=1)
k_theory_b = np.arange(0,np.max(degrees_b),1)
ax2.loglog(k_theory_b,Proportions_b,color='red',label = 'Theoretical calculations',linewidth=1, alpha=1)





#p=1, q =0.25
#Binning
mybins_c= np.unique(np.logspace(np.log10(1),np.log10(np.max(degrees_c)+1), num=20, endpoint=True, base=10.0, dtype=int))
degrees_c=np.ndarray.flatten(np.array(degrees_c))  # To make a flat list(array) out of a list(array) of lists(arrays)
hist_c = np.histogram(degrees_c, bins=mybins_c)
pdf_c =hist_c[0]/np.sum(hist_c[0])#normalize histogram --for pdf
box_sizes=mybins_c[1:]-mybins_c[:-1]  # size of boxes
pdf_c = pdf_c/box_sizes   # Divide pdf by its box size

#bins = Midpoints of distribution
mid_points_c=np.power(10, np.log10(mybins_c[:-1]) + (np.log10(mybins_c[1:]-1)-np.log10(mybins_c[:-1]))/2)
#mid_points_g=mid_points_g. astype(int)  # Mid_points_g as integers
#Linear binning
#ncounts_c, bins_c = np.histogram(degrees_c,bins=np.arange(1,np.max(degrees_c),1))
#pdf_c1=ncounts_c/np.sum(ncounts_c)  #normalize histogram --for pdf -probability density function       
#ax3.scatter(bins_c[:-1],pdf_c1,color='green', label = 'Numerical calculations', alpha=0.5) 
ax3.loglog(mid_points_c,pdf_c,color= 'blue',label ='Numerical calculations', linewidth=1, alpha=1)
k_theory_c = np.arange(0,np.max(degrees_c),1)
ax3.loglog(k_theory_c,Proportions_c,color='red',label = 'Theoretical calculations',linewidth=1, alpha=1)




#p=0.75, q =1
#Binning
mybins_d= np.unique(np.logspace(np.log10(1),np.log10(np.max(degrees_d)+1), num=30, endpoint=True, base=10.0, dtype=int))
degrees_d=np.ndarray.flatten(np.array(degrees_d))  # To make a flat list(array) out of a list(array) of lists(arrays)
hist_d = np.histogram(degrees_d, bins=mybins_d)
pdf_d =hist_d[0]/np.sum(hist_d[0])#normalize histogram --for pdf
box_sizes=mybins_d[1:]-mybins_d[:-1]  # size of boxes
pdf_d = pdf_d/box_sizes   # Divide pdf by its box size

#bins = Midpoints of distribution
mid_points_d=np.power(10, np.log10(mybins_d[:-1]) + (np.log10(mybins_d[1:]-1)-np.log10(mybins_d[:-1]))/2)
#mid_points_g=mid_points_g. astype(int)  # Mid_points_g as integers
#Linear binning
#ncounts_d, bins_d = np.histogram(degrees_d,bins=np.arange(1,np.max(degrees_d),1))
#pdf_d1=ncounts_d/np.sum(ncounts_d)  #normalize histogram --for pdf -probability density function       
#ax4.scatter(bins_d[:-1],pdf_d1,color='green', label = 'Numerical calculations', alpha=0.5) 
ax4.loglog(mid_points_d,pdf_d,color= 'blue',label ='Numerical calculations', linewidth=1, alpha=1)
k_theory_d = np.arange(0,np.max(degrees_d),1)
ax4.loglog(k_theory_d,Proportions_d,color='red',label = 'Theoretical calculations',linewidth=1, alpha=1)



#p=0.25, q =1
#Binning
mybins_e= np.unique(np.logspace(np.log10(1),np.log10(np.max(degrees_e)+1), num=20, endpoint=True, base=10.0, dtype=int))
degrees_e=np.ndarray.flatten(np.array(degrees_e))  # To make a flat list(array) out of a list(array) of lists(arrays)
hist_e = np.histogram(degrees_e, bins=mybins_e)
pdf_e =hist_e[0]/np.sum(hist_e[0])#normalize histogram --for pdf
box_sizes=mybins_e[1:]-mybins_e[:-1]  # size of boxes
pdf_e = pdf_e/box_sizes   # Divide pdf by its box size

#bins = Midpoints of distribution
mid_points_e=np.power(10, np.log10(mybins_e[:-1]) + (np.log10(mybins_e[1:]-1)-np.log10(mybins_e[:-1]))/2)
#mid_points_g=mid_points_g. astype(int)  # Mid_points_g as integers
#Linear binning
#ncounts_e, bins_e = np.histogram(degrees_e,bins=np.arange(1,np.max(degrees_e),1))
#pdf_e1=ncounts_e/np.sum(ncounts_e)  #normalize histogram --for pdf -probability density function       
#ax5.scatter(bins_e[:-1],pdf_e1,color='green', label = 'Numerical calculations', alpha=0.5) 
#ax5.loglog(mid_points_e,pdf_e,color= 'blue',label ='Numerical calculations', linewidth=1, alpha=1)

ax5.loglog(mid_points_e,pdf_e,color= 'blue',label ='Numerical calculations', linewidth=1, alpha=1)

k_theory_e = np.arange(0,np.max(degrees_e),1)
ax5.loglog(k_theory_e,Proportions_e,color='red',label = 'Theoretical calculations',linewidth=1, alpha=1)




#p=0.5, q =0.5
#Binning
mybins_f= np.unique(np.logspace(np.log10(1),np.log10(np.max(degrees_f)+1), num=30, endpoint=True, base=10.0, dtype=int))
degrees_f=np.ndarray.flatten(np.array(degrees_f))  # To make a flat list(array) out of a list(array) of lists(arrays)
hist_f = np.histogram(degrees_f, bins=mybins_f)
pdf_f =hist_f[0]/np.sum(hist_f[0])#normalize histogram --for pdf
box_sizes=mybins_f[1:]-mybins_f[:-1]  # size of boxes
pdf_f = pdf_f/box_sizes   # Divide pdf by its box size

#bins = Midpoints of distribution
mid_points_f=np.power(10, np.log10(mybins_f[:-1]) + (np.log10(mybins_f[1:]-1)-np.log10(mybins_f[:-1]))/2)
#mid_points_g=mid_points_g. astype(int)  # Mid_points_g as integers

#Linear binning
#ncounts_f, bins_f = np.histogram(degrees_f,bins=np.arange(1,np.max(degrees_f),1))
#ncounts, bins = np.histogram(degrees_g,bins=mybins_g)
#pdf_f1=ncounts_f/np.sum(ncounts_f)  #normalize histogram --for pdf -probability density function       
#ax6.scatter(bins_f[:-1],pdf_f1,color='green', label = 'Numerical calculations', alpha=0.5) 
ax6.loglog(mid_points_f,pdf_f,color= 'blue',label ='Numerical calculations', linewidth=1, alpha=1)
k_theory_f = np.arange(0,np.max(degrees_f),1)
ax6.loglog(k_theory_f,Proportions_f,color='red',label = 'Theoretical calculations',linewidth=1, alpha=1)








for ax in fig.get_axes():
    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
	    label.set_fontsize(20)
    ax.set(xlabel='Degree $k$', ylabel='Frequency')
    ax.label_outer()  # Set axis scales outer

 ## Label panels    
ax1.text(0.3, 0.9, 'A', fontsize=20,fontweight='bold')
ax2.text(0.3, 0.9, 'B', fontsize=20,fontweight='bold')
ax3.text(0.3, 0.9, 'C', fontsize=20,fontweight='bold')
ax4.text(0.3, 0.9, 'D', fontsize=20,fontweight='bold')
ax5.text(0.3, 0.9, 'E', fontsize=20,fontweight='bold')
ax6.text(0.3, 0.9, 'F', fontsize=20,fontweight='bold')

plt.xlim([1,100000])  # Limiting x axis
plt.ylim([0.0000000000000001,1])  # Limiting x axis
#plt.savefig('Degree_distribution_loglog_model3.pdf')
#
#












### THIS BIINING APPROACH WAS DISCARDED B'SE IT SEEMS NOT TO GIVE CORRECT SLOPE OF DATA 
#fig, ((ax1, ax2), (ax3, ax4),(ax5, ax6)) = plt.subplots(3,2,figsize=(16, 18), sharex=True, sharey=True, squeeze= True)
#plt.rcParams['font.size'] = '20'
#
###p=1, q =1
###Binning
#mybins_a= np.unique(np.logspace(np.log10(1),np.log10(np.max(degrees_a)+1), num=25, endpoint=True, base=10.0, dtype=int))
#degrees_a=np.ndarray.flatten(np.array(degrees_a))  # To make a flat list(array) out of a list(array) of lists(arrays)
#hist_a = np.histogram(degrees_a, bins=mybins_a)
#pdf_a =hist_a[0]/np.sum(hist_a[0])#normalize histogram --for pdf
#ax1.loglog(mybins_a[:-1],pdf_a,color= 'blue', label="Numerical calculations",linewidth=1.5)
##pdf_theory_a=np.zeros(len(mybins_a)-1)
#pdf_theory_a=np.zeros(len(mybins_a[:-1]))
#
#for i,j in enumerate(mybins_a[:-1]):
#    #print(j-1)
#    #print(Proportions_a[j:mybins_a[i+1]])
#    #It the sum of all the proportions in each range of bins because it is 
#    # for example, all nodes with 10 to 12 connection, 13 to 18 and so on.
#    #This makes theory same as simulation.
#    pdf_theory_a[i]=np.sum(Proportions_a[j:mybins_a[i+1]])        
#ax1.loglog(mybins_a[:-1],pdf_theory_a,color='red',label = 'Theoretical calculations', alpha=0.5)
#
#
#mybins_b= np.unique(np.logspace(np.log10(1),np.log10(np.max(degrees_b)+1), num=20, endpoint=True, base=10.0, dtype=int))
#hist_b = np.histogram(degrees_b, bins=mybins_b)
#pdf_b =hist_b[0]/np.sum(hist_b[0])  #normalize histogram --for pdf
#ax2.loglog(mybins_b[:-1],pdf_b,color= 'blue', label="Numerical calculations",linewidth=1.5)
#
#pdf_theory_b=np.zeros(len(mybins_b[:-1]))
#for i,j in enumerate(mybins_b[:-1]):
#    pdf_theory_b[i]=np.sum(Proportions_b[j:mybins_b[i+1]])
#    
#    
#ax2.loglog(mybins_b[:-1],pdf_theory_b,color='red',label = 'Theoretical calculations', alpha=0.5)
#
#mybins_c= np.unique(np.logspace(np.log10(1),np.log10(np.max(degrees_c)+1), num=20, endpoint=True, base=10.0, dtype=int))
#hist_c = np.histogram(degrees_c, bins=mybins_c)
#pdf_c =hist_c[0]/np.sum(hist_c[0])  #normalize histogram --for pdf
#ax3.loglog(mybins_c[:-1],pdf_c,color= 'blue', label="Numerical calculations",linewidth=1.5)
#
#pdf_theory_c=np.zeros(len(mybins_c[:-1]))
#for i,j in enumerate(mybins_c[:-1]):
#    pdf_theory_c[i]=np.sum(Proportions_c[j:mybins_c[i+1]])
#    
#    
#ax3.loglog(mybins_c[:-1],pdf_theory_c,color='red',label = 'Theoretical calculations', alpha=0.5)
#
#mybins_d= np.unique(np.logspace(np.log10(1),np.log10(np.max(degrees_d)+1), num=20, endpoint=True, base=10.0, dtype=int))
#hist_d = np.histogram(degrees_d, bins=mybins_d)
#pdf_d =hist_d[0]/np.sum(hist_d[0])  #normalize histogram --for pdf
#ax4.loglog(mybins_d[:-1],pdf_d,color= 'blue', label="Numerical calculations",linewidth=1.5)
#
#pdf_theory_d=np.zeros(len(mybins_d[:-1]))
#for i,j in enumerate(mybins_d[:-1]):
#    pdf_theory_d[i]=np.sum(Proportions_d[j:mybins_d[i+1]])
#    
#    
#ax4.loglog(mybins_d[:-1],pdf_theory_d,color='red',label = 'Theoretical calculations', alpha=0.5)
#mybins_e= np.unique(np.logspace(np.log10(1),np.log10(np.max(degrees_e)+1), num=20, endpoint=True, base=10.0, dtype=int))
#hist_e = np.histogram(degrees_e, bins=mybins_e)
#pdf_e =hist_e[0]/np.sum(hist_e[0])  #normalize histogram --for pdf
#ax5.loglog(mybins_e[:-1],pdf_e,color= 'blue', label="Numerical calculations",linewidth=1.5)
#
#pdf_theory_e=np.zeros(len(mybins_e[:-1]))
#for i,j in enumerate(mybins_e[:-1]):
#    pdf_theory_e[i]=np.sum(Proportions_e[j:mybins_e[i+1]])        
#ax5.loglog(mybins_e[:-1],pdf_theory_e,color='red',label = 'Theoretical calculations', alpha=0.5)
#mybins_f= np.unique(np.logspace(np.log10(1),np.log10(np.max(degrees_f)+1), num=20, endpoint=True, base=10.0, dtype=int))
#hist_f = np.histogram(degrees_f, bins=mybins_f)
#pdf_f =hist_f[0]/np.sum(hist_f[0])  #normalize histogram --for pdf
#ax6.loglog(mybins_f[:-1],pdf_f,color= 'blue', label="Numerical calculations",linewidth=1.5)
#
#pdf_theory_f=np.zeros(len(mybins_f[:-1]))
#for i,j in enumerate(mybins_f[:-1]):
#    pdf_theory_f[i]=np.sum(Proportions_f[j:mybins_f[i+1]])    
#ax6.loglog(mybins_f[:-1],pdf_theory_f,color='red',label = 'Theoretical calculations', alpha=0.5)
#for ax in fig.get_axes():
#    for label in (ax.get_xticklabels() + ax.get_yticklabels()):
#	    label.set_fontsize(20)
#    ax.set(xlabel='Degree $k$', ylabel='Frequency')
#    ax.label_outer()  # Set axis scales outer
#
# ## Label panels    
#ax1.text(2.5, 0.7, 'A', fontsize=20,fontweight='bold')
#ax2.text(3.3, 0.9, 'B', fontsize=20,fontweight='bold')
#ax3.text(2.5, 0.7, 'C', fontsize=20,fontweight='bold')
#ax4.text(3.3, 0.9, 'D', fontsize=20,fontweight='bold')
#ax5.text(2.5, 0.7, 'E', fontsize=20,fontweight='bold')
#ax6.text(3.3, 0.9, 'F', fontsize=20,fontweight='bold')
#
#plt.xlim([1,100000])  # Limiting x axis
#plt.ylim([0.00000000001,1])  # Limiting x axis
##plt.savefig('Degree_distribution_loglog_model3.pdf')
#
