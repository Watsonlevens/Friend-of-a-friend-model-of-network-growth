#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 10:41:50 2021
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
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

##Numerical results. Opening the degrees
##100 simulations, N=100,000
#########################################################################################################################
#openfile = open('all_all_deg_g.pkl', 'rb') # N = 100000 for 1000 runs
#degrees_g = pickle.load(openfile)          # p = 0, q = 1
##########################################################################################################################

# 5 simulations, N=1000,000
########################################################################################################################
openfile = open('all_all_deg_g1.pkl', 'rb') # N = 1000000 for 5 runs
degrees_g = pickle.load(openfile)          # p = 0, q = 1
#########################################################################################################################

#Theoretical Results
#########################################################################################################################
openfile = open('Proportions_g1.pkl', 'rb') # p = 0, q = 1
Proportions_g = pickle.load(openfile) # Size of proportions = np.max(degrees_g)
#########################################################################################################################

#Log bins . We want 1,2,3 because special case. Then we have 4-6, 7-10
mybins_g= np.unique(np.logspace(np.log10(1),np.log10(np.max(degrees_g)+1), num=30, endpoint=True, base=10.0, dtype=int))
degrees_g=np.ndarray.flatten(np.array(degrees_g))  # To make a flat list(array) out of a list(array) of lists(arrays)
hist_g = np.histogram(degrees_g, bins=mybins_g)
pdf_g =hist_g[0]/np.sum(hist_g[0])#normalize histogram --for pdf

box_sizes=mybins_g[1:]-mybins_g[:-1]
pdf_g = pdf_g/box_sizes

#bins = Midpoints of distribution
mid_points_g=np.power(10, np.log10(mybins_g[:-1]) + (np.log10(mybins_g[1:]-1)-np.log10(mybins_g[:-1]))/2)
#mid_points_g=mid_points_g. astype(int)  # Mid_points_g as integers
#Linear binning
#Different bins for the same data
ncounts, bins = np.histogram(degrees_g,bins=np.arange(1,np.max(degrees_g),1))
#ncounts, bins = np.histogram(degrees_g,bins=mybins_g)
pdf_g1=ncounts/np.sum(ncounts)  #normalize histogram --for pdf -probability density function       


####compute alpha by maximum likelhood
x_min =10
some_deg = degrees_g[degrees_g>=x_min]    ##remove zeros from np.array 
n = len(np.array(some_deg)) 

alpha = 1 + n/sum(np.log((some_deg)/ (x_min-0.5))) ##calculate alpha
print('alpha by maximum likelihood alpha:',alpha)


# Calculate alpha by linear regression
pdf = pdf_g 
biggerthan0= pdf>0                               
k = mid_points_g
k=k[biggerthan0]
#print(k)
pdf=pdf[biggerthan0]
lk=np.log(k)
#print(lk)
lP=np.log(pdf)  
#slope, intercept, r_value, p_value, std_err = stats.linregress(lk,lP)          
model = pd.DataFrame()
model = model.assign(lP=lP)
model = model.assign(lk=lk)                
#Slope determined by linear regression
model_fit=smf.ols(formula='lP ~ lk ', data=model).fit()
   # print(model_fit.summary())                 
# Fit data       
b=model_fit.params # Compute slope and y-intercept (y = mx +c)
slope = b[1]
alpha = -b[1]
print('alpha by linear regression:',alpha)


#Plotting the data
fig,ax=plt.subplots(1,figsize=(8, 7), sharex=True, sharey=True, squeeze= True)
plt.rcParams['font.size'] = '12'

ax.scatter(bins[:-1],pdf_g1,color='green', label = 'Numerical calculations', alpha=0.8) 
ax.plot(mid_points_g,pdf_g,color= 'blue',label ='Numerical calculations', linewidth=1, alpha=1)
#ax.plot(mybins_g[:-1],pdf_theory_g,color='red',label = 'Theoretical calculations',linewidth=1, alpha=0.5)
#ax.plot(mybins_g[:-1],pdf_simuation_g,color='black',label = 'Theoretical calculations',linewidth=1, alpha=0.5)
k_theory = np.arange(0,len(Proportions_g),1)
ax.plot(k_theory,Proportions_g,color='red',label = 'Theoretical calculations',linewidth=1, alpha=1)


# x-y scale 
plt.xscale('log')
plt.yscale('log')

ax.set_xlabel('Degree $k$')
ax.set_ylabel('Frequency')
plt.xlim([1,np.max(degrees_g)+25367])  # Limiting x axis
plt.ylim([0.000000000000001,1])  # Limiting x axis # Limiting x axis
#plt.savefig('loglog_combined_model3.pdf')



