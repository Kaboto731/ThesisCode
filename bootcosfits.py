#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 21:00:56 2020

@author: manuel
"""
import numpy as np
from cosfit2 import bootcos
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable


def bootcosinefit(timesteps,nangles,proheatinput,midheatinput,supheatinput,proforce,midforce,supforce,graphtitle):
        #Window = the window of time that is averaged over
        window = 5
        #showing single neuron activity
        total = timesteps
        #full neuron behavior pronated
        for i in range(total):
            timechosen=i
            prosneuron = np.zeros([nangles,100,1])
            #neurons at the 88th time step
            proaveragedneuron = np.mean(proheatinput[:,timechosen:timechosen+window,:],axis=1)
            prosneuron =np.transpose(proheatinput[:,timechosen,:]) #np.transpose(proaveragedneuron) 
            #posneuron is neuronsx angles
            #The force at the 88th time step
            prosneuron2 = proforce[:,timechosen]
            bootcos(prosneuron2,prosneuron,timechosen,0)
            print("Completed",i+1,"/",total)
        cmap = plt.get_cmap('jet_r')
        sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0,100))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Timestep', rotation=270,labelpad=25)
        if (graphtitle==True):
            plt.title("Pronated")
        plt.show()
        #full neuron behavior midrange
        for i in range(total):
            timechosen=i
            midsaveragedneuron = np.transpose(np.mean(midheatinput[:,timechosen:timechosen+window,:],axis=1))
            midsneuron = np.transpose(midheatinput[:,timechosen,:])
            midsneuron2 = midforce[:,timechosen]
            bootcos(midsneuron2,midsneuron,timechosen,0)
            print("Completed",i+1,"/",total)
        cmap = plt.get_cmap('jet_r')
        sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0,100))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Timestep', rotation=270,labelpad=25)
        if (graphtitle==True):
            plt.title("Midrange")
        plt.show()
        #full neuron behavior -supinated
        for i in range(total):
            timechosen=i
            supsaveragedneuron = np.transpose(np.mean(supheatinput[:,timechosen:timechosen+window,:],axis=1))
            supsneuron = np.transpose(supheatinput[:,timechosen,:])
            supsneuron2 = supforce[:,timechosen]
            bootcos(supsneuron2,supsneuron,timechosen,0)
            print("Completed",i+1,"/",total)
        cmap = plt.get_cmap('jet_r')
        sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0,100))
        sm.set_array([])
        cbar = plt.colorbar(sm)
        cbar.set_label('Timestep', rotation=270,labelpad=25)
        if (graphtitle==True):
            plt.title("Supinated")
        plt.show()
        
def neuronbootcosinefit(proforce,midmyneuron,supmyneuron,timesteps,window,proheatinput,midheatinput,supheatinput,promyneuron,supforce,midforce,nangles):
    total =timesteps
    totalavg=timesteps-window
    #pronated
    for i in range(totalavg):
        timechosen=i
        #average over the time of the window for each posture 
        proaveragedneuron = np.transpose(np.mean(proheatinput[:,timechosen:timechosen+window, promyneuron],axis=1))
        proaveragedneuron= np.reshape(proaveragedneuron,[1,16])
        prosneuron = np.zeros([nangles,100,1])
        #posneuron is neuronsx angles
        #The force at the 88th time step
        aprosneuron2 = proforce[:,timechosen]
        bootcos(aprosneuron2,proaveragedneuron,timechosen,0)
        print("Completed",i+1,"/",total)
    cmap = plt.get_cmap('jet_r')
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0,100))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Timestep', rotation=270,labelpad=25)
    plt.show()
    
    
      
    #non-average windowing
    for i in range(total):
        timechosen=i
        aprosneuron = np.transpose(proheatinput[:,timechosen,promyneuron])
        aprosneuron = np.reshape(aprosneuron,[1,16])
        #posneuron is neuronsx angles
        #The force at the 88th time step
        aprosneuron2 = proforce[:,timechosen]
        bootcos(aprosneuron2,aprosneuron,timechosen,0)
        print("Completed",i+1,"/",total)
    cmap = plt.get_cmap('jet_r')
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0,100))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Timestep', rotation=270,labelpad=25)
    plt.show()
    #midrange
    #MIDRANGE
    for i in range(totalavg):
        timechosen=i
        #average over the time of the window for each posture 
        midaveragedneuron = np.transpose(np.mean(midheatinput[:,timechosen:timechosen+window, midmyneuron],axis=1))
        midaveragedneuron= np.reshape(midaveragedneuron,[1,16])
        #posneuron is neuronsx angles
        #The force at the 88th time step
        amidsneuron2 = midforce[:,timechosen]
        bootcos(amidsneuron2,midaveragedneuron,timechosen,0)
        print("Completed",i+1,"/",total)
    cmap = plt.get_cmap('jet_r')
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0,100))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Timestep', rotation=270,labelpad=25)
    plt.show()
    for i in range(total):
        timechosen=i
        amidsneuron = np.transpose(midheatinput[:,timechosen,midmyneuron])
        amidsneuron = np.reshape(amidsneuron,[1,16])
        amidsneuron2 = midforce[:,timechosen]
        bootcos(amidsneuron2,amidsneuron,timechosen,0)
        print("Completed",i+1,"/",total)
    cmap = plt.get_cmap('jet_r')
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0,100))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Timestep', rotation=270,labelpad=25)
    plt.show()
    
    
    
    
    #supinated
    for i in range(totalavg):
        timechosen=i
        #average over the time of the window for each posture 
        supaveragedneuron = np.transpose(np.mean(supheatinput[:,timechosen:timechosen+window, supmyneuron],axis=1))
        supaveragedneuron= np.reshape(supaveragedneuron,[1,16])
        #posneuron is neuronsx angles
        #The force at the 88th time step
        asupsneuron2 = supforce[:,timechosen]
        bootcos(asupsneuron2,supaveragedneuron,timechosen,0)
        print("Completed",i+1,"/",total)
    cmap = plt.get_cmap('jet_r')
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0,100))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Timestep', rotation=270,labelpad=25)
    plt.show()
    
    
    
    for i in range(total):
        timechosen=i
        asupsneuron = np.transpose(supheatinput[:,timechosen,supmyneuron])
        asupsneuron = np.reshape(asupsneuron,[1,16])
        asupsneuron2 = supforce[:,timechosen]
        bootcos(asupsneuron2,asupsneuron,timechosen,0)
        print("Completed",i+1,"/",total)
    cmap = plt.get_cmap('jet_r')
    sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(0,100))
    sm.set_array([])
    cbar = plt.colorbar(sm)
    cbar.set_label('Timestep', rotation=270,labelpad=25)
    plt.show()