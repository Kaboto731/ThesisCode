#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 00:21:09 2020

@author: manuel
"""
import numpy as np
#from keras import backend

import matplotlib.pyplot as plt

def plotnngraphs(patience,proforce,midforce,supforce,v,myweights,history,vslice,graphtitle,n,expectedoutput2,goat,timer,tester):

        #if (heatplot == True ):
        #    sb.heatmap(myoutput[1])
        
        print(history.keys())
        
        #Since our loss is MSE these calculate the same thing
        #plt.plot(history.history['loss'])
        #plt.title("Loss Value" )
        #plt.xlabel("epoch")
        #plt.ylabel("Value")
        #plt.show()
        #plt.plot(history.history['mean_absolute_error'])
        #we use validation in order to not include the error of the droppedout neuron
        plt.plot(np.sqrt(history['val_mean_squared_error'][:-patience-1]))
        if (graphtitle == True):
            plt.title("Validation Error" )
        plt.xlabel("epoch")
        plt.ylabel("Validation RMSE")
        plt.ylim([0,0.5])
        plt.savefig('./Thesis/ThesisChapter/ThesisImg/epochRMSE.pdf',bbox_inches='tight')
        plt.show()
        plt.plot(history['val_loss'][:-patience-1])
        plt.plot(history['loss'][:-patience-1])
        plt.ylim([0,1])
        if (graphtitle== True):
            plt.title('RNN-model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        #plt.axhline(y=0, color='k', linestyle='-')
        plt.legend(['validation loss', 'training loss'], loc='upper right')
        plt.savefig('./Thesis/ThesisChapter/ThesisImg/loss.pdf',bbox_inches='tight')
        plt.close()
        
        plt.plot(v[tester])
        if (graphtitle== True):
            plt.title("Visual input" )
        plt.xlabel("visual input")
        plt.ylabel("Activation")
        plt.show()
        plt.hist(myweights)
        if (graphtitle == True):
            plt.title("MI activation weights" )
        plt.xlabel("MI neuron weights")
        plt.ylabel("Frequency")
        plt.show()
        plt.plot(proforce[tester,:,0],proforce[tester,:,1],'b-',expectedoutput2[tester,:,0],expectedoutput2[tester,:,1],'r-')
        if (graphtitle== True):
            plt.title("X-Force versus Y-force")
        plt.xlabel("X-Force")
        plt.ylabel("Y-Force")
        plt.xlim([-1.2,1.2])
        plt.ylim([-1.2,1.2])
        plt.gca().set_aspect('equal', adjustable='box')
        plt.legend(['Output','Expected Output'])
        plt.savefig('./Thesis/ThesisChapter/ThesisImg/xy.pdf',bbox_inches='tight')
        plt.show()
        fig, axs = plt.subplots(2,sharex=True,figsize=(10, 5)) 
        axs[0].plot(timer,proforce[tester,:,0],'b-',timer,expectedoutput2[tester,:,0],'r-')
        if (graphtitle== True):
            axs[0].set_title("X-Force versus time")
        axs[0].set_ylim([-1.2,1.2])
        axs[0].set_ylabel("X-Force")
        axs[0].legend(['Output','Expected Output'])
        axs[0].axvline(x=timer[int(goat[tester])], linewidth = 2, color = 'g')
        axs[0].axvline(x=4.5, linewidth=2, color='y')
        axs[1].plot(timer,proforce[tester,:,1],'b-',timer,expectedoutput2[tester,:,1],'r-')
        if (graphtitle== True):
            axs[1].set_title("Y-Force versus time")
        axs[1].set_xlabel("Time")
        axs[1].set_ylim([-1.2,1.2])
        axs[1].set_ylabel("Y-Force")
        #axs[1].legend(['Output','Expected Output'])
        axs[1].axvline(x=timer[int(goat[tester])], linewidth = 2, color = 'g')
        axs[1].axvline(x=4.5, linewidth=2, color='y')
        fig.savefig("xtyt.pdf", bbox_inches='tight')
        plt.show()