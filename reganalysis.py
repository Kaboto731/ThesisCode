#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:46:35 2020

@author: manuel
"""
import numpy as np
import matplotlib.pyplot as plt
from nangleunitrepresentation import nanglescircle
from keras import backend
import gc

def myanalysis(timesteps,epo,examples,numRNNneurons,m1a,mua,nangle,qvals,wvals):
    endpoint = np.zeros([qvals,wvals,2])
    m1aregs = np.zeros([qvals])
    muaregs = np.zeros([wvals])
    for q in range(qvals):
        m1aregs[q] = m1a*q*25
        for w in range(wvals):
            muaregs[w] = mua*w*25
            scores=nanglescircle(timesteps,epo,examples,numRNNneurons,m1a*q*25,mua*w*25,nangle)
            endpoint[q,w] = scores
            backend.clear_session()
            gc.collect()
            
            print("my progress:"+str(q*(wvals)+w+1)+"/"+str(qvals*wvals))
    #saving our stuff in our file once we are done
    np.savetxt('endpoints.csv', endpoint, delimiter=',', fmt='%f')
    endpoint2 = np.loadtxt("endpoints.csv", delimiter=',')
    endpoint2 = np.array(endpoint2)
    fig, ax = plt.subplots(figsize=(10,10))
    ax.imshow(endpoint2)
    ax.set_xticks(np.arange(qvals))
    ax.set_yticks(np.arange(wvals))
    
    muaregs.tostring()
    m1aregs.tostring()
    ax.set_xticklabels(m1aregs)
    ax.set_yticklabels(muaregs)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right",
     rotation_mode="anchor")
    for i in range(qvals):
        for j in range(wvals):
            text = ax.text(j, i, endpoint2[i, j],
                           ha="center", va="center", color="k")
    #ax.set_title("Heatmap of MSE error with different activation levels")
    ax.set_xlabel("M1 Activation Reglarization")
    ax.set_ylabel("Muscle Activation Regularization")
    fig.tight_layout()
    plt.show()            