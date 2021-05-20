#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 15 17:45:38 2020

@author: manuel
"""

from math import pi
import math as ma
from scipy import optimize
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import random
import numpy.matlib
import builtins
from matplotlib.cm import ScalarMappable
def fix_angles(angs):
    angs_fixed = np.zeros(len(angs))
    for i in range(len(angs)):
        angs_fixed[i] = angs[i]%(2*pi)
        if (angs_fixed[i]>=pi):
            angs_fixed[i]=angs_fixed[i]-2*pi
    return angs_fixed
def fix_angles_val(angs):
    angs_fixed = 0
    angs_fixed = angs%(2*pi)
    if (angs_fixed>=pi):
        angs_fixed=angs_fixed-2*pi
    return angs_fixed

def cos_fit_helper(xscos,yscos):
    #Builds a cosine fit of the aproxximated yscos given the xscos
    
    #how many numbers we have to fit the angles
    fit_num_neurons = yscos.shape[0]
    #pdsin = np.zeros([fit_num_neurons]) r2 is single val
    #r2sin = np.zeros([fit_num_neurons]) pd is single val
    c = np.zeros(3)
    X = np.transpose(xscos)
    ys_biased = yscos
    #the bias of each neuron
    bias = np.mean(ys_biased,axis=0)
    ys_unbiased = ys_biased-bias
    ys_unbiased.reshape((fit_num_neurons,1))
    #the lsq_coeffs of x and y force
    lsq_coeffs = np.matmul(np.linalg.inv(np.matmul(np.transpose(X),X)),np.matmul(np.transpose(X),ys_unbiased))
    #the theta of each time step
    thetas = np.arctan2(xscos[1,:],xscos[0,:])
    X_norm = np.sqrt(builtins.sum(np.square(X)))
    #The bias of the neuron
    c[0] = bias
    c[1] = np.linalg.norm(lsq_coeffs[0:1])*np.mean(X_norm)
    #the PD of the neuron
    c[2] = np.arctan2(lsq_coeffs[1], lsq_coeffs[0])
    #pd is theta
    pdsin = c[2]
    #Calculating residuals    
    residuals = yscos - np.matmul(X,lsq_coeffs)
    r2sin = 1 - builtins.sum(np.square(residuals))/ builtins.sum((yscos-np.mean(yscos))**2)
    return pdsin, c,r2sin
def plot_cos_fit(plotxs,plotys,coeffs,timechosen):
    cmap = plt.get_cmap('jet_r')
    color = cmap(float(timechosen)/100)
    mythetas = np.arctan2(plotxs[:,1],plotxs[:,0])
    thetas = np.arange(min(mythetas),max(mythetas),0.01)
    fit_ys = coeffs[0] + coeffs[1]*np.cos(coeffs[2]-thetas)
    plt.plot(mythetas*180.0/np.pi,plotys,'o',c=color)
    plt.plot(thetas*180.0/np.pi,fit_ys,c=color)
    plt.ylabel("activity level")
    plt.xlabel("Degrees")
    
    #timearray = np.linspace(0,1,100)
    #plt.imshow(timearray, cmap=cmap)
def bootcos(xs,ys,timechosen,scatter):
    #firing rate ys
    ys = np.array(ys)
    #force vector
    xs = np.array(xs)
    #ys is num_neurons X num_pts
    num_neurons = ys.shape[0]
    num_pts = ys.shape[1]
    coeffs = np.zeros([num_neurons,3])
    
    stability_thresh= 25*(ma.pi/180)
    
    inds = np.linspace(1,num_pts,num_pts)
    #number of times for resampling
    num_samples = 150
    #preffered direction for each neuron
    pds = np.zeros(num_neurons)
    #pds for subsamplings each neuron
    pds_multi = np.zeros([num_neurons,num_samples])
    #R^2's multi for each sample
    r2s = np.zeros([num_neurons, num_samples])
    #our random samples each with 100pts
    rand_sample = np.zeros([num_samples, num_pts])
    #for each neuron we take a random subsampling and putting it into cos_fit_helper
    for neuron in range(num_neurons):
        for samples in range(num_samples):
            # Sample with replacement
            sample_inds = random.choices(inds,k=num_pts)
            
            sample_inds = np.array(sample_inds)
            #integer for indices
            sample_inds = sample_inds.astype(int)
            #shifting from 1-x to 0-(x-1)
            sample_inds2 = sample_inds-1
            #saving indices of the sample
            rand_sample[samples] = sample_inds2
            #taking the sample indices of the resulting force
            xs_sub = xs[sample_inds2,:]
            #taking a random sample of the neuron
            ys_sub = ys[neuron,sample_inds2]
            #pds_multi[:, samples], coeffs[neuron,:], r2s[:,samples] = cos_fit_helper(xs_sub,ys_sub[0,:],-1)
            pds_multi[neuron, samples], coeffs[neuron,:], r2s[neuron,samples] = cos_fit_helper(np.transpose(xs_sub),np.transpose(ys_sub))
    dropp_for_stab = 0
    dropp_for_r2 = 0
    dropp_for_excessive_dom =0


    for neuron in range(num_neurons):
        
       pds_multi[neuron,:]= pds_multi[neuron,:]+np.pi
       pds_multi[neuron,:] = np.sort(pds_multi[neuron,:])
       
       low = pds_multi[neuron, ma.floor(num_samples*0.025)]
       high = pds_multi[neuron, ma.floor(num_samples*0.975)]
       
       rad_range = fix_angles_val(abs(high-low))
       stable = False
       if (abs(rad_range)> 2*stability_thresh):
            #neuron is not stable
            pds[neuron] = float('NaN')
            coeffs[neuron,:] = float('NaN')
            
            dropp_for_stab = dropp_for_stab+1
       else:
            #neuron is stable
            stable=True
            mean_pd = (low+high)/2
            dists = abs(pds_multi[neuron,:]-mean_pd)
            Y = np.sort(dists)
            I = np.argsort(dists)
            Iinner = np.zeros(ma.floor(len(I)/2))
            
            Iinner = I[0:ma.floor(len(I)/2)]
            
            max_r2 = max(r2s[neuron,Iinner])
            r2I = np.argmax(r2s[neuron,Iinner])           
            r2I = I[r2I]
           #choosing the best out of the subsampling
            xs_sub = xs[rand_sample[r2I].astype(int),:]
            ys_sub = ys[neuron,rand_sample[r2I].astype(int)]
            pds[neuron], coeffs[neuron,:],r2 = cos_fit_helper(np.transpose(xs_sub),np.transpose(ys_sub))
            
            
            #R2 value is crazy
            if (r2 < 0.2 or np.isnan(r2) or np.isinf(r2)):#(r2 < 0.2 or np.isnan(r2) or np.isinf(r2)):
                pds[neuron] = float('NaN')
                coeffs[neuron,:] = float('NaN')
                dropp_for_r2 = dropp_for_r2+1
                
            else:
            #R squared is acceptable
                if ((coeffs[neuron,0]+coeffs[neuron,1])/np.percentile(ys_sub[:],90)>1.5):
                    #the neurons base activity and oscillation is based off of outliers
                    pds[neuron] = float('NaN')
                    coeffs[neuron,:] = float('NaN')
                    dropp_for_excessive_dom = dropp_for_excessive_dom+1
                
                else:
                    #if all else holds we plot it
                    if (scatter != 1):
                        plot_cos_fit(xs_sub,ys_sub, coeffs[neuron,:],timechosen)
      
    #print("Dropped for Stability:",dropp_for_stab)
    #print("Dropped for R2",dropp_for_r2)
    #print("Dropped for Excessive Domain:",dropp_for_excessive_dom)
    passed = np.argwhere(~np.isnan(pds))                
    if (dropp_for_excessive_dom+dropp_for_r2+dropp_for_stab<num_neurons):
        print("time:",timechosen)
        print("neurons:",passed)
        print("total:",dropp_for_excessive_dom+dropp_for_r2+dropp_for_stab, "/",num_neurons)
    if (scatter ==1):
           return pds[:],coeffs[:,1]