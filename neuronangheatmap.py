#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 20:53:50 2020

@author: manuel
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sb

def fullpds(numRNNneurons,timer,prothetamax,midthetamax,supthetamax, graphtitle,runindex):
        a4_dims = (10.27, 8.27)
        fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = a4_dims,gridspec_kw={'width_ratios': [10,10,10], 
                       'wspace': 0.1})
        for i in range(numRNNneurons):
            neuron = i
            #vmaxer=3
            #proheatneuron = proheatinput[:,:,neuron]
            #midheatneuron = midheatinput[:,:,neuron]
            #supheatneuron = supheatinput[:,:,neuron]
            prothetamaxneuron = prothetamax[neuron,:]*180/np.pi
            midthetamaxneuron = midthetamax[neuron,:]*180/np.pi
            supthetamaxneuron = supthetamax[neuron,:]*180/np.pi
            #prothetamaxneuron = prothetamaxneuron[~numpy.isnan(prothetamaxneuron)]
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Target Angle (Degrees)")
            ax1.set_title("Pronated")
            ax1.set_xlim([0,5])
            ax1.plot(timer,prothetamaxneuron)
            #sb.lineplot(x=timer*20, y=prothetamaxneuron,ax=ax1,ci=None)
            ax2.set_xlabel("Time (s)")
            ax2.plot(timer, midthetamaxneuron)
            #sb.lineplot(x=timer*20, y=prothetamaxneuron,ax=ax2,ci=None)
            ax2.set_xlim([0,5])
            ax2.set_title("Midrange")
            ax2.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelright=False,
            labelleft = False)
            ax3.plot(timer,supthetamaxneuron)
            #sb.lineplot(x=timer*20, y=prothetamaxneuron,ax=ax3,ci=None)
            ax3.set_title("Supinated")
            ax3.set_xlabel("Time (s)")
            ax3.set_xlim([0,5])
            ax3.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelright=False,
            labelleft = False)
        plt.savefig('./fullpdsr'+str(runindex)+'.pdf')
def fullneuron(proheatinput,midheatinput,supheatinput,theta2,numRNNneurons,history,timer,prothetamax,midthetamax,supthetamax, graphtitle,runindex):
    fullheatpro = np.full_like(proheatinput[:,:,0],0)
    fullheatmid =  np.full_like(proheatinput[:,:,0],0)
    fullheatsup = np.full_like(proheatinput[:,:,0],0)

    fullheatpro = np.sum(proheatinput,axis=2)
    fullheatmid = np.sum(midheatinput,axis=2)
    fullheatsup = np.sum(supheatinput,axis=2)
    fullall = fullheatpro+fullheatmid+fullheatsup
    a4_dims = (10.27, 8.27)
    fig, (ax1, ax2,ax3,ax4) = plt.subplots(1,4, figsize = a4_dims,gridspec_kw={'width_ratios': [10,10,10, 1], 
                               'wspace': 0.1})
    propd = pd.DataFrame(fullheatpro, index=np.round(theta2*180/np.pi,2),columns = np.round(timer,0))
    midpd = pd.DataFrame(fullheatmid, index=np.round(theta2*180/np.pi,2),columns = np.round(timer,0))
    suppd = pd.DataFrame(fullheatsup, index=np.round(theta2*180/np.pi,2),columns = np.round(timer,0))
    allpds = pd.DataFrame(fullall, index=np.round(theta2*180/np.pi,2),columns = np.round(timer,0))
    sb.heatmap(propd,cbar = False,ax=ax1,vmin=0, xticklabels=np.arange(0,6,1),cmap="YlGnBu" )
    ax1.axvline(x=30, color="g")
    ax1.axvline(x=90, color="y")
    #ax1.plot(timer*20,maxpropds, color = 'r')
    ax1.set_title("Pronated")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Target Angle (Degrees)")
    ax1.set_xticks([0,20,40,60,80,100])#("0","0.5","1","1.5","2","2.5","3","3.5","4","4.5","5")
    sb.heatmap(midpd,cbar = False,yticklabels=False,ax=ax2,vmin=0, xticklabels=np.arange(0,6,1),cmap="YlGnBu" )
    ax2.axvline(x=30, color="g")
    ax2.axvline(x=90, color="y")
    #ax2.plot(timer*20,maxmidpds, color = 'r')
    ax2.set_xticks([0,20,40,60,80,100])
    ax2.set_title("Midrange")
    ax2.set_xlabel("Time (s)")    
    sb.heatmap(suppd,cbar = True,cbar_ax = ax4,yticklabels=False, ax=ax3,vmin=0, xticklabels=np.arange(0,6,1),cmap="YlGnBu")
    ax3.axvline(x=30, color="g")
    ax3.axvline(x=90, color="y")
    #ax3.plot(timer*20,maxsuppds, color = 'r')
    ax3.set_xticks([0,20,40,60,80,100])
    ax3.set_title("Supinated")
    ax3.set_xlabel("Time (s)") 
    fig.savefig('heatmapplots3/allneuronplot'+'r'+str(runindex)+'.pdf',bbox_inches='tight')
    plt.close()
    a4_dims = (10.27, 8.27)
    fig2, axs = plt.subplots(figsize=a4_dims)
    sb.heatmap(allpds,cbar = True,ax=axs,vmin=0, xticklabels=np.arange(0,6,1),cmap="YlGnBu")
    axs.axvline(x=30, color="g")
    axs.axvline(x=90, color="y")
    #ax3.plot(timer*20,maxsuppds, color = 'r')
    axs.set_xticks([0,20,40,60,80,100])            
    axs.set_xlabel("Time (s)")
    axs.set_ylabel("Target Angle (Degrees)")
    fig2.savefig('./heatmapplots3/allsumneuronplot.pdf',bbox_inches='tight')
    plt.close()
         #propd = pd.DataFrame(proheatneuron, index=np.round(theta2*180/np.pi,2),columns = np.round(timer,1))# columns = np.arange(0,5,0.5)

    #fullheatpropd = pd.DataFrame(fullheatpro, index=np.round(theta2*180/np.pi,2),columns = np.round(timer,0))
    
def neuronheatmap(proheatinput,midheatinput,supheatinput,theta2,numRNNneurons,history,timer,prothetamax,midthetamax,supthetamax, graphtitle,runindex):

        a4_dims = (10.27, 8.27)
        for i in range(numRNNneurons):
            neuron = i
            vmaxer=3
            proheatneuron = proheatinput[:,:,neuron]
            midheatneuron = midheatinput[:,:,neuron]
            supheatneuron = supheatinput[:,:,neuron]
            prothetamaxneuron = prothetamax[neuron,:]*180/np.pi
            midthetamaxneuron = midthetamax[neuron,:]*180/np.pi
            supthetamaxneuron = supthetamax[neuron,:]*180/np.pi
            fig, (ax1, ax2,ax3,ax4) = plt.subplots(1,4, figsize = a4_dims,gridspec_kw={'width_ratios': [10,10,10, 1], 
                                   'wspace': 0.1})
             #propd = pd.DataFrame(proheatneuron, index=np.round(theta2*180/np.pi,2),columns = np.round(timer,1))# columns = np.arange(0,5,0.5)
            propd = pd.DataFrame(proheatneuron, index=np.round(theta2*180/np.pi,2),columns = np.round(timer,0))
            prodata = propd.idxmax()
            sb.heatmap(propd,cbar = False ,ax=ax1,vmin=0, xticklabels=np.arange(0,6,1),cmap="YlGnBu")#, cmap="YlGnBu"
            
            #fig.tight_layout()
            ax1.axvline(x=30, color="g")
            ax1.axvline(x=90, color="y")
            #ax1.plot(timer*20,maxpropds, color = 'r')
            ax1.set_title("Pronated")
            ax1.set_xlabel("Time (s)")
            ax1.set_ylabel("Target Angle (Degrees)")
            ax1.set_xticks([0,20,40,60,80,100])#("0","0.5","1","1.5","2","2.5","3","3.5","4","4.5","5")
            #ax1.set_ylabel("Target Direction (Deg)")
            ax5 = ax1.twinx()
            ax5 = plt.gca()
            ax5.set_ylim([157,-180])
            ax5.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelright=False) # labels along the bottom edge are off
            ax5.plot(timer*20,prothetamaxneuron,color = 'r')
            #sb.lineplot(x=timer*20, y=prothetamaxneuron,ax=ax5,ci=None,color='red') 
             
            midpd = pd.DataFrame(midheatneuron, index=np.round(theta2*180/np.pi,2), columns = np.round(timer,0))
            #maxmidpds = np.max(midpd,axis=0)
            # plt.subplot(1,3,2)
            #fig.tight_layout()
            middata = midpd.idxmax()
            sb.heatmap(midpd,cbar = False,yticklabels=False,ax=ax2,vmin=0, xticklabels=np.arange(0,6,1),cmap="YlGnBu" )#,cmap="YlGnBu"
            ax6 = ax2.twinx()
            ax6 = plt.gca()
            ax6.set_ylim([157,-180])
            ax6.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelright=False) # labels along the bottom edge are off
            sb.lineplot(x=timer*20, y=midthetamaxneuron,ax=ax6,ci=None,color='red')
            ax2.axvline(x=30, color="g")
            ax2.axvline(x=90, color="y")
            #ax2.plot(timer*20,maxmidpds, color = 'r')
            ax2.set_xticks([0,20,40,60,80,100])
            ax2.set_title("Midrange")
            ax2.set_xlabel("Time (s)")
             #ax2.set_ylabel("Target Direction (deg)")
             #plt.show()
             
            suppd = pd.DataFrame(supheatneuron, index=np.round(theta2*180/np.pi,2), columns = np.round(timer,0))
            #maxsuppds = np.max(supheatneuron,axis=0)
            # plt.subplot(1,3,3)
            #fig.tight_layout()
            supdata = suppd.idxmax()
            ax7 = ax3.twinx()
            ax7 = plt.gca()
            ax7.set_ylim([157,-180])
            ax7.tick_params(
            axis='y',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            left=False,      # ticks along the bottom edge are off
            right=False,         # ticks along the top edge are off
            labelright=False) # labels along the bottom edge are off
            
            sb.heatmap(suppd,cbar = True,cbar_ax = ax4,yticklabels=False, ax=ax3,vmin=0, xticklabels=np.arange(0,6,1),cmap="YlGnBu")#,cmap="YlGnBu"
            sb.lineplot(x=timer*20, y=supthetamaxneuron,ax=ax7,ci=None,color='red')
            ax3.axvline(x=30, color="g")
            ax3.axvline(x=90, color="y")
            #ax3.plot(timer*20,maxsuppds, color = 'r')
            ax3.set_xticks([0,20,40,60,80,100])
            ax3.set_title("Supinated")
            ax3.set_xlabel("Time (s)")
            
            fig.savefig('heatmapplots3/neuronplot'+str(neuron)+'r'+str(runindex)+'.pdf',bbox_inches='tight')
            plt.close()