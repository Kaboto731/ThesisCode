#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Fri Sep 13 12:56:02 2019

@author: manuel
"""
from keras import backend
import matplotlib.pyplot as plt
import gc
import numpy as np
#custom functions
from pddomscatter2 import pddomscatter
from plotactivationfunction import plotactivfunc #works
#from fullunitrepresentation import fullcircle - not used only nangle points
from nngraphplotter import plotnngraphs
from visgrapher import plotvisgraphs
from locevalplot import locationplot
#from plotendpoints import plotendpointMSE - wrote better code below
#from reganalysis import myanalysis
from posnanalgeunitrepresentationcopy import nanglesposcirlce
from fvafrmse import plotendpointrmse
from fvafrmse import savefvaf
from fvafplotter import plotsavedfvafs
from neuronangheatmap import neuronheatmap
from neuronangheatmap import fullneuron
from bootcosfits2 import bootcosinefit
from bootcosfits2 import neuronbootcosinefit
from stableregplotter import regplots
from stableregplotter import neuronregplots
from neuronangheatmap import fullpds
from cosfit2 import bootcos
import os
import pickle
from pdcleaner import cleanpds
#from neuralnet import tf_eslu
#Switch timer back to 0.05
#Switch goat to 20-40
#Make examples 5 or greater up to 20
#Plotting--------------------------------------------------
plt.rcParams.update({'font.size': 12})

timesteps = 100
epo = 2500
examples = 30000
numRNNneurons = 100
graphtitle = False
m1a = 0.000000001
mua = 0.000000001
kernm1a = 0.000001
kernmua = 0.0000001
patience = 100
nangle = 16
tester = 0
qvals = 4 
wvals = 3
val = 4
runindex = 4
#runindex = 1
prothetamaxs = np.zeros([numRNNneurons,timesteps])
propdmaxs = np.zeros([numRNNneurons,timesteps])
midthetamaxs = np.zeros([numRNNneurons,timesteps])
midpdmaxs = np.zeros([numRNNneurons,timesteps])
supthetamaxs = np.zeros([numRNNneurons,timesteps])
suppdmaxs = np.zeros([numRNNneurons,timesteps])

plotvisgraphs(2500, 100)
plotactivfunc(False)

#mymodel,mymodel2,history,v,vslice,handmat,myweights,output2,goat,timer,pcount,mcount,scount,orientation,theta,holdperiod = fullcircle(timesteps,epo,examples,numRNNneurons,savennfile,loadnnfile,m1a,mua)
expectedoutput2,scores, mymodel, mymodel2,v,handmat, myweights, history,vslice,goat,output2,theta,pcount,mcount,scount,orientation,timer,theta2,vslicereshapepro,reshapedhandmatpro,proheatinput,proheatoutput,proforce,vslicereshapemid,reshapedhandmatmid,midheatinput,midheatoutput,midforce,vslicereshapesup,reshapedhandmatsup,supheatinput,supheatoutput,supforce,holdperiod=nanglesposcirlce(kernm1a,kernmua,patience,timesteps,epo,examples,numRNNneurons,m1a,mua,nangle,runindex)
for j in range(numRNNneurons):
    midmyneuron=j
    supmyneuron =midmyneuron
    promyneuron = midmyneuron
    
    window = 2
    neuronbootcosinefit(proforce,midmyneuron,supmyneuron,timesteps,window,proheatinput,midheatinput,supheatinput,promyneuron,supforce,midforce,nangle,prothetamaxs,propdmaxs,midthetamaxs,midpdmaxs,supthetamaxs,suppdmaxs,False,runindex)
neuronheatmap(proheatinput,midheatinput,supheatinput,theta2,numRNNneurons,history,timer,prothetamaxs,midthetamaxs,supthetamaxs,graphtitle,runindex)
fullneuron(proheatinput, midheatinput, supheatinput, theta2, numRNNneurons, history, timer, prothetamaxs, midthetamaxs, supthetamaxs, graphtitle, runindex)
#locationplot(theta2+np.pi)
#bootcosinefit(timesteps,nangle,proheatinput,midheatinput,supheatinput,proforce,midforce,supforce,graphtitle,prothetamaxs,propdmaxs,midthetamaxs,midpdmaxs,supthetamaxs,suppdmaxs)
fullpds(numRNNneurons,timer,prothetamaxs,midthetamaxs,supthetamaxs, graphtitle,runindex)
for i in range(numRNNneurons):
    midmyneuron=i
    supmyneuron =midmyneuron
    promyneuron = midmyneuron
    
    window = 2
    neuronbootcosinefit(proforce,midmyneuron,supmyneuron,timesteps,window,proheatinput,midheatinput,supheatinput,promyneuron,supforce,midforce,nangle,prothetamaxs,propdmaxs,midthetamaxs,midpdmaxs,supthetamaxs,suppdmaxs)
PIK = "pickle.dat"
data = [proheatinput,midheatinput,supheatinput,theta2,numRNNneurons,history,timer,prothetamaxs,midthetamaxs,supthetamaxs,graphtitle]
with open(PIK, "wb") as f:
    pickle.dump(len(data), f)
    for value in data:
        pickle.dump(value, f)
data2 = []
with open(PIK, "rb") as f:
    for _ in range(pickle.load(f)):
        data2.append(pickle.load(f))

#neuronheatmap(data2[0],data2[1],data2[2],data2[3],data2[4],data2[5],data2[6],data2[7],data2[8],data2[9],data2[10])
#fullpds(data2[4],data2[6],data2[7],data2[8],data2[9], data2[10])
for i in range(1):
    runindex=i
    expectedoutput2,scores, mymodel, mymodel2,v,handmat, myweights, history,vslice,goat,output2,theta,pcount,mcount,scount,orientation,timer,theta2,vslicereshapepro,reshapedhandmatpro,proheatinput,proheatoutput,proforce,vslicereshapemid,reshapedhandmatmid,midheatinput,midheatoutput,midforce,vslicereshapesup,reshapedhandmatsup,supheatinput,supheatoutput,supforce,holdperiod=nanglesposcirlce(kernm1a,kernmua,patience,timesteps,epo,examples,numRNNneurons,m1a,mua,nangle,runindex)
    for j in range(numRNNneurons):
        midmyneuron=j
        supmyneuron =midmyneuron
        promyneuron = midmyneuron
        
        window = 2
        neuronbootcosinefit(proforce,midmyneuron,supmyneuron,timesteps,window,proheatinput,midheatinput,supheatinput,promyneuron,supforce,midforce,nangle,prothetamaxs,propdmaxs,midthetamaxs,midpdmaxs,supthetamaxs,suppdmaxs,False,runindex)
    neuronheatmap(proheatinput,midheatinput,supheatinput,theta2,numRNNneurons,history,timer,prothetamaxs,midthetamaxs,supthetamaxs,graphtitle,runindex)
    print("Finished"+str(i+1)+'/'+str(5))
    fullpds(numRNNneurons,timer,prothetamaxs,midthetamaxs,supthetamaxs, graphtitle,runindex)
    gc.collect()
    backend.clear_session()
    
    
    
#plots individual run information
plotnngraphs(patience,proforce,midforce,supforce,v,myweights,history,vslice,graphtitle,numRNNneurons,expectedoutput2,goat,timer,tester)
plotendpointrmse(proforce,midforce,supforce, nangle,holdperiod,numRNNneurons,timesteps,expectedoutput2,graphtitle)
tester = 0
barfvaf = True
rmseplots = True




regplots(timesteps,epo,examples,numRNNneurons,nangle,kernm1a,kernmua,patience)



# mean of 20 plots RMSE and Loss
possiblehistory = np.zeros([20,500])
possibleloss = np.zeros([20,500])
possibletrainloss = np.zeros([20,500])
for i in range(20):
    runindex = i
    expectedoutput2,scores, mymodel, mymodel2,v,handmat, myweights, history,vslice,goat,output2,theta,pcount,mcount,scount,orientation,timer,theta2,vslicereshapepro,reshapedhandmatpro,proheatinput,proheatoutput,proforce,vslicereshapemid,reshapedhandmatmid,midheatinput,midheatoutput,midforce,vslicereshapesup,reshapedhandmatsup,supheatinput,supheatoutput,supforce,holdperiod=nanglesposcirlce(kernm1a,kernmua,patience,timesteps,epo,examples,numRNNneurons,m1a,mua,nangle,runindex)
    possibleloss[i,:len(history['val_loss'][:-patience-1])] = history['val_loss'][:-patience-1]
    possibletrainloss[i,:len(history['loss'][:-patience-1])] = history['loss'][:-patience-1]
    possiblehistory[i,:len(history['val_mean_squared_error'][:-patience-1])] = history['val_mean_squared_error'][:-patience-1]
    gc.collect()
    backend.clear_session()
meanpossibleloss = np.mean(possibleloss,axis=0)
nozeropossibleloss=meanpossibleloss[meanpossibleloss !=0]
meanpossibletrainloss = np.mean(possibletrainloss,axis=0)
nozeropossibletrainloss=meanpossibletrainloss[meanpossibletrainloss !=0]
sqrtpossiblehistory = np.sqrt(possiblehistory)
meanpossibleRMSE = np.mean(sqrtpossiblehistory,axis=0)
nozeroRMSE = meanpossibleRMSE[meanpossibleRMSE !=0]
plt.plot(nozeroRMSE)
plt.xlabel("epoch")
plt.ylabel("Validation RMSE")
plt.ylim([0,0.5])
plt.savefig('./Thesis/ThesisChapter/ThesisImg/epochRMSE.pdf',bbox_inches='tight')
plt.plot(nozeropossibletrainloss)
plt.plot(nozeropossibleloss)
plt.ylim([0,1])
if (graphtitle== True):
    plt.title('RNN-model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
#plt.axhline(y=0, color='k', linestyle='-')
plt.legend(['validation loss', 'loss'], loc='upper right')
plt.savefig('./Thesis/ThesisChapter/ThesisImg/loss.pdf',bbox_inches='tight')
plt.close()


for i in range(10):
    runindex = i
    expectedoutput2,scores, mymodel, mymodel2,v,handmat, myweights, history,vslice,goat,output2,theta,pcount,mcount,scount,orientation,timer,theta2,vslicereshapepro,reshapedhandmatpro,proheatinput,proheatoutput,proforce,vslicereshapemid,reshapedhandmatmid,midheatinput,midheatoutput,midforce,vslicereshapesup,reshapedhandmatsup,supheatinput,supheatoutput,supforce,holdperiod=nanglesposcirlce(kernm1a,kernmua,patience,timesteps,epo,examples,numRNNneurons,m1a,mua,nangle,runindex)
    plt.plot(np.sqrt(history['val_mean_squared_error'][:-patience-1]))
    if (graphtitle == True):
        plt.title("Validation Error" )
    plt.xlabel("epoch")
    plt.ylabel("RMSE")
    plt.ylim([0,0.5])


#FVAF of 20 plots saved to a file
for i in range(20):
    runindex = i
    savennfile = False
    loadnnfile = False
    timesteps = 100
    epo = 2500
    examples = 30000
    numRNNneurons = 100
    m1a = 0.00000001
    mua = 0.000000001
    nangle = 16
    expectedoutput2,scores, mymodel, mymodel2,v,handmat, myweights, history,vslice,goat,output2,theta,pcount,mcount,scount,orientation,timer,theta2,vslicereshapepro,reshapedhandmatpro,proheatinput,proheatoutput,proforce,vslicereshapemid,reshapedhandmatmid,midheatinput,midheatoutput,midforce,vslicereshapesup,reshapedhandmatsup,supheatinput,supheatoutput,supforce,holdperiod=nanglesposcirlce(kernm1a,kernmua,patience,timesteps,epo,examples,numRNNneurons,m1a,mua,nangle,runindex)
    savefvaf(nangle,expectedoutput2,proforce,midforce,supforce)
    plotsavedfvafs()
    
    # os.remove('modelneurons100m1a999999999kernm1a1000000mua10000000000kernmua10000000r1.h5')
    # os.remove('weightsneurons100m1a999999999kernm1a1000000mua10000000000kernmua10000000r1.h5')
    # os.remove('myhistneurons100m1a999999999kernm1a1000000mua10000000000kernmua10000000r1.json')
    gc.collect()
    backend.clear_session()
    print("counter:",i+1)
    

#MSE by neuron now handled by neuronregplots
# neuronvec = [25,50,75,100,125]
# numruns=5
# myendingMSE = np.zeros([len(neuronvec),numruns])
# for i in range(len(neuronvec)):
#     for j in range(numruns):
#         runindex=j
#         expectedoutput2,scores, mymodel, mymodel2,v,handmat, myweights, history,vslice,goat,output2,theta,pcount,mcount,scount,orientation,timer,theta2,vslicereshapepro,reshapedhandmatpro,proheatinput,proheatoutput,proforce,vslicereshapemid,reshapedhandmatmid,midheatinput,midheatoutput,midforce,vslicereshapesup,reshapedhandmatsup,supheatinput,supheatoutput,supforce,holdperiod=nanglesposcirlce(kernm1a,kernmua,patience,timesteps,epo,examples,neuronvec[i],m1a,mua,nangle,runindex)
#         myendingMSE[i][j]=history['val_mean_squared_error'][-patience - 1]
#         print(myendingMSE[i][j])
#         gc.collect()
#         backend.clear_session()
# #saving the neuron MSE
# f = open('neuronMSE.pkl', 'wb')
# pickle.dump(myendingMSE, f)
# f.close
# print ('data saved')   
# f = open('neuronMSE.pkl', 'rb')
# obj = pickle.load(f)
# f.close()
# print ('data loaded')
# plt.xlabel('Number of neurons')
# plt.ylabel('Validation MSE')     
# plt.plot(neuronvec,np.mean(myendingMSE,axis=1))    
# plt.savefig('neuronMSEplot.pdf',bbox_inches='tight')

# plotendpointMSE(timesteps,epo,examples,numRNNneurons,savennfile,loadnnfile,neuronvec,graphtitle)


#Scatterplot and historgram of PD and DoM

for i in range(20):
    runindex = i
    savennfile = False
    loadnnfile = False
    timesteps = 100
    epo = 2500
    examples = 30000
    numRNNneurons = 100
    nangle = 16
    expectedoutput2,scores, mymodel, mymodel2,v,handmat, myweights, history,vslice,goat,output2,theta,pcount,mcount,scount,orientation,timer,theta2,vslicereshapepro,reshapedhandmatpro,proheatinput,proheatoutput,proforce,vslicereshapemid,reshapedhandmatmid,midheatinput,midheatoutput,midforce,vslicereshapesup,reshapedhandmatsup,supheatinput,supheatoutput,supforce,holdperiod=nanglesposcirlce(kernm1a,kernmua,patience,timesteps,epo,examples,numRNNneurons,m1a,mua,nangle,runindex)
    pddomscatter(nangle,proheatinput,midheatinput,supheatinput,proforce,midforce,supforce,timesteps,numRNNneurons,holdperiod)
    gc.collect()
    backend.clear_session()
    print("counter:",i+1)
#cleans/deletes the PD files 
#cleanpds()
#boot cosine fits of all activity
bootcosinefit(timesteps,nangle,proheatinput,midheatinput,supheatinput,proforce,midforce,supforce,graphtitle,prothetamaxs,propdmaxs,midthetamaxs,midpdmaxs,supthetamaxs,suppdmaxs)


#myanalysis(holdperiod,timesteps,epo,examples,numRNNneurons,m1a,mua,nangle,qvals,wvals)
        



