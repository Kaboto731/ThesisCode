#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:55:43 2020

@author: manuel
"""
import numpy as np
from cosfit3 import bootcosscatter
import math as ma
import matplotlib.pyplot as plt
import pickle
from scipy import stats
from itertools import chain 
def pddomscatter(nangles,proheatinput,midheatinput,supheatinput,proforce,midforce,supforce,timesteps,numRNNneurons,holdperiod):
        #pronated PD and Depth of modulation
        numRNNneurons = len(midheatinput[0,0,:])
        propds = np.zeros([holdperiod,numRNNneurons])
        procoeffs = np.zeros([holdperiod,numRNNneurons])
        #midrange PD and Depth of modulation
        midpds = np.zeros([holdperiod,numRNNneurons])
        midcoeffs = np.zeros([holdperiod,numRNNneurons])
        #supinated PD and Depth of modulation
        suppds = np.zeros([holdperiod,numRNNneurons])
        supcoeffs = np.zeros([holdperiod,numRNNneurons])
        # For every time step we get stable neurons with pd and dom in pronated
        for i in range(holdperiod):
            timechosen=timesteps-holdperiod+i
            prosneuron = np.zeros([nangles,1,timesteps])
            #neurons at the 88th time step
            prosneuron = np.transpose(proheatinput[:,timechosen,:])
            #posneuron is neuronsx angles
            #The force at the 88th time step
            prosneuron2 = proforce[:,timechosen]
            propds[i], procoeffs[i]= bootcosscatter(prosneuron2,prosneuron,timechosen)

            midsneuron = np.zeros([nangles,1,timesteps])
            #neurons at the 88th time step
            midsneuron = np.transpose(midheatinput[:,timechosen,:])
            #posneuron is neuronsx angles
            #The force at the 88th time step
            midsneuron2 = midforce[:,timechosen]
            midpds[i], midcoeffs[i]= bootcosscatter(midsneuron2,midsneuron,timechosen)
            
            supsneuron = np.zeros([nangles,1,timesteps])
            #neurons at the 88th time step
            supsneuron = np.transpose(supheatinput[:,timechosen,:])
            #posneuron is neuronsx angles
            #The force at the 88th time step
            supsneuron2 = supforce[:,timechosen]
            suppds[i], supcoeffs[i]= bootcosscatter(supsneuron2,supsneuron,timechosen)

        #plot on a scatterplot PD shift vs Depth of modulation
            
        prosupshift=suppds-propds 
        promidshift=midpds- propds
        shiftmidcoeffs = (midcoeffs-procoeffs)/((procoeffs+midcoeffs)/2)
        shiftsupcoeffs = (supcoeffs-procoeffs)/((procoeffs+supcoeffs)/2)
        #plt.scatter(prosupshift*180/ma.pi,shiftsupcoeffs)
        #plt.xlabel("pronated-supinated pdshift")
        #plt.ylabel("depth of modulation supinated")
        #plt.show()flattenedprosupshift
        #plt.scatter(promidshift*180/ma.pi,shiftmidcoeffs)
        #plt.xlabel("pronated-midrange pdshift")
        #plt.ylabel("depth of modulation midrange")        
        #plt.show()
        
        nonanprosupshift = prosupshift[~np.isnan(prosupshift)]*180/ma.pi
        nonanshiftsupcoeffs  = shiftsupcoeffs[~np.isnan(prosupshift)]
        
        nonanpromidshift = promidshift[~np.isnan(promidshift)]*180/ma.pi
        nonanshiftmidcoeffs  = shiftmidcoeffs[~np.isnan(promidshift)]
        
        
       # plt.scatter(nonanprosupshift,nonanshiftsupcoeffs)
       # plt.show()
       # plt.scatter(nonanpromidshift,nonanshiftmidcoeffs)
       # plt.show()
        #run='' being orig neuron scat with 1e-9 m1a
        #run2 being muscles of run 1
        #run 3 is the neuron scat with 8e-9 m1a
        myrun = 3
        with open("procoeffs"+str(myrun), 'ab') as fp:
            pickle.dump(procoeffs,fp)
            fp.close()        
        with open("propds"+str(myrun), 'ab') as fp:
            pickle.dump(propds,fp)
            fp.close()
        with open("midpds"+str(myrun), 'ab') as fp:
            pickle.dump(midpds,fp)
            fp.close()
        with open("suppds"+str(myrun), 'ab') as fp:
            pickle.dump(suppds,fp)
            fp.close()
        with open("supcoeffs"+str(myrun), 'ab') as fp:
            pickle.dump(nonanshiftsupcoeffs,fp)
            fp.close()
        with open("promidshift"+str(myrun), 'ab') as fp:
            pickle.dump(nonanpromidshift,fp)
            fp.close()
        with open("prosupshift"+str(myrun), 'ab') as fp:
            pickle.dump(nonanprosupshift,fp)
            fp.close()
        with open("midcoeffs"+str(myrun), 'ab') as fp:
            pickle.dump(nonanshiftmidcoeffs,fp)
            fp.close()
        
        readprosupshift = []
        readprosupcoeffs = []
        readpromidshift = []
        readpromidcoeffs = []
        with open("prosupshift"+str(myrun), 'rb') as fr:
            try:
                while True:
                    readprosupshift.append(pickle.load(fr))
            except EOFError:
                pass  
        fr.close()
        with open("promidshift"+str(myrun), 'rb') as fr:
            try:
                while True:
                    readpromidshift.append(pickle.load(fr))
            except EOFError:
                pass 
        fr.close()
        with open("supcoeffs"+str(myrun), 'rb') as fr:
            try:
                while True:
                    readprosupcoeffs.append(pickle.load(fr))
            except EOFError:
                pass  
        fr.close()
        with open("midcoeffs"+str(myrun), 'rb') as fr:
            try:
                while True:
                    readpromidcoeffs.append(pickle.load(fr))
            except EOFError:
                pass  
        fr.close()
        print("num of times:",len(readprosupshift))
        flattenedprosupshift = list(chain.from_iterable(readprosupshift))
        flattenedsupcoeffs = list(chain.from_iterable(readprosupcoeffs))
        flattenedpromidshift = list(chain.from_iterable(readpromidshift))
        flattenedpromidcoeffs = list(chain.from_iterable(readpromidcoeffs))
        flattenedsupcoeffs = np.array(flattenedsupcoeffs)
        flattenedpromidcoeffs  = np.array(flattenedpromidcoeffs)
        flattenedprosupshift = np.array(flattenedprosupshift)
        flattenedpromidshift = np.array(flattenedpromidshift)
        #T Test of difference
        t, p = stats.ttest_ind(flattenedprosupshift,flattenedpromidshift)
        t2,p2 = stats.ttest_ind(flattenedpromidcoeffs,flattenedsupcoeffs)
        
        fig, (ax1,ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [10,10], 
                       'wspace': 0.4}) 
        ax1.scatter(flattenedprosupshift,flattenedsupcoeffs*100,s=5) 
        ax1.set_xlabel("PD Shift (supinated-pronated) Degrees")
        ax1.set_ylabel("Percent of DoM Shift (supinated-pronated)")
        ax1.set_ylim([-225,225])
        ax1.set_xlim([-45,45])
        #plt.ylim([-400,400])
        ax2.hist(flattenedprosupshift,bins=30)
        ax2.axvline(x=np.mean(flattenedprosupshift), color="r")
        ax2.set_ylabel('Frequency')
        ax2.set_xlabel('PD shift (Degrees)')
        ax2.set_xlim([-45,45])
        plt.savefig('./Thesis/ThesisChapter/ThesisImg/muscscathistprosupshift2.pdf',bbox_inches='tight')
        for i in range(len(flattenedpromidshift)):
            if (flattenedpromidshift[i]<-180):
                flattenedpromidshift[i]=flattenedpromidshift[i]+360
            if (flattenedpromidshift[i]>180):
                flattenedpromidshift[i]=flattenedpromidshift[i]-360
        fig, (ax3, ax4) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [10,10], 
                       'wspace': 0.4}) 
        ax3.scatter(flattenedpromidshift,flattenedpromidcoeffs*100,s=5) 
        ax3.set_xlabel("PD Shift (midrange-pronated) Degrees")
        ax3.set_ylabel("Percent of DoM Shift (midrange-pronated)") 
        ax3.set_ylim([-225,225])
        ax3.set_xlim([-45,45])
        #plt.ylim([-400,400])
        ax4.hist(flattenedpromidshift,bins=30)
        ax4.set_ylabel('Frequency')
        ax4.set_xlabel('PD shift (Degrees)')
        ax4.axvline(x=np.mean(flattenedpromidshift), color="r")
        ax4.set_xlim([-45,45])
        plt.savefig('./Thesis/ThesisChapter/ThesisImg/muscscathistpromidshift2.pdf',bbox_inches='tight')
        #plt.show()