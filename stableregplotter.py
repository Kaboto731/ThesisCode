#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 27 09:27:50 2020

@author: manuel
"""
import numpy as np
from posnanalgeunitrepresentationcopy import nanglesposcirlce
from cosfit2 import bootcos
import matplotlib.pyplot as plt
import gc
from keras import backend
from scipy import stats
import pickle
def regplots(timesteps,epo,examples,numRNNneurons,nangle,kernm1a,kernmua,patience):
    runs = 20
    qvals=5
    myendingMSEm1a = np.zeros([runs,qvals])
    myendingMSEmua = np.zeros([runs,qvals])
    y = np.zeros([runs,qvals])
    ym = np.zeros([runs,qvals])
    ys = np.zeros([runs,qvals])
    y2 = np.zeros([runs,qvals])
    ym2 = np.zeros([runs,qvals])
    ys2 = np.zeros([runs,qvals])
    muas = np.zeros([qvals])
    m1as = np.zeros([qvals])
    muas[0] = m1as[0] = 0.000000001
    for i in range(runs):
        runindex = i
        for j in range(qvals):
                mua = muas[0]
                m1a = m1as[j] = 0.000000001*2**j
                expectedoutput2,scores, mymodel, mymodel2,v,handmat, myweights, history,vslice,goat,output2,theta,pcount,mcount,scount,orientation,timer,theta2,vslicereshapepro,reshapedhandmatpro,proheatinput,proheatoutput,proforce,vslicereshapemid,reshapedhandmatmid,midheatinput,midheatoutput,midforce,vslicereshapesup,reshapedhandmatsup,supheatinput,supheatoutput,supforce,holdperiod=nanglesposcirlce(kernm1a,kernmua,patience,timesteps,epo,examples,numRNNneurons,m1a,mua,nangle,runindex)
                myendingMSEm1a[i][j]=history['val_mean_squared_error'][-patience - 1]
                for k in range(holdperiod):
                    q = timesteps-holdperiod+k
                    prosneuron = np.zeros([nangle,100,1])
                    #neurons at the 88th time step
                    prosneuron =np.transpose(proheatinput[:,q,:]) #np.transpose(proaveragedneuron) 
                    midsneuron =np.transpose(midheatinput[:,q,:])
                    supsneuron =np.transpose(supheatinput[:,q,:])
                    #posneuron is neuronsx angles
                    #The force at the 88th time step
                    prosneuron2 = proforce[:,q]
                    midsneuron2 = midforce[:,q]
                    supsneuron2 = supforce[:,q]
                    pds,coeffs = bootcos(prosneuron2,prosneuron,q,1)
                    pds2,coeffs2  = bootcos(midsneuron2,midsneuron,q,1)
                    pds3,coeffs3  = bootcos(supsneuron2,supsneuron,q,1)
                    #remove NaNs and sum for each run and vals
                    y[runindex,j]=y[runindex,j]+len(pds[~np.isnan(pds)])
                    ym[runindex,j] = ym[runindex,j]+len(pds2[~np.isnan(pds2)])
                    ys[runindex,j] = ys[runindex,j]+len(pds3[~np.isnan(pds3)])
                gc.collect()
                backend.clear_session()
        m1a = 0.000000001
        for j in range(qvals):
                m1a =m1as[0]
                mua  = muas[j] = 0.000000001*2**j
                expectedoutput2,scores, mymodel, mymodel2,v,handmat, myweights, history,vslice,goat,output2,theta,pcount,mcount,scount,orientation,timer,theta2,vslicereshapepro,reshapedhandmatpro,proheatinput,proheatoutput,proforce,vslicereshapemid,reshapedhandmatmid,midheatinput,midheatoutput,midforce,vslicereshapesup,reshapedhandmatsup,supheatinput,supheatoutput,supforce,holdperiod=nanglesposcirlce(kernm1a,kernmua,patience,timesteps,epo,examples,numRNNneurons,m1a,mua,nangle,runindex)
                myendingMSEmua[i][j]=history['val_mean_squared_error'][-patience - 1]
                for k in range(holdperiod):
                    q = timesteps-holdperiod+k
                    prosneuron = np.zeros([nangle,100,1])
                    #neurons at the 88th time step
                    prosneuron =np.transpose(proheatinput[:,q,:]) #np.transpose(proaveragedneuron) 
                    midsneuron =np.transpose(midheatinput[:,q,:])
                    supsneuron =np.transpose(supheatinput[:,q,:])
                    #posneuron is neuronsx angles
                    #The force at the 88th time step
                    prosneuron2 = proforce[:,q]
                    midsneuron2 = midforce[:,q]
                    supsneuron2 = supforce[:,q]
                    pds,coeffs = bootcos(prosneuron2,prosneuron,q,1)
                    pds2,coeffs2  = bootcos(midsneuron2,midsneuron,q,1)
                    pds3,coeffs3  = bootcos(supsneuron2,supsneuron,q,1)
                    #remove NaNs and sum for each run and vals
                    y2[runindex,j]=y2[runindex,j]+len(pds[~np.isnan(pds)])
                    ym2[runindex,j] = ym2[runindex,j]+len(pds2[~np.isnan(pds2)])
                    ys2[runindex,j] = ys2[runindex,j]+len(pds3[~np.isnan(pds3)])
                gc.collect()
                backend.clear_session()
        mua = 0.000000001
    file1 = 'neuronMSEm1a.pkl'  
    file2 = 'neuronMSEmua.pkl'
    file3 = 'myys.pkl'
    file4 = 'myy2s.pkl'
    file5 = 'myyms.pkl'
    file6 = 'myyms2.pkl'
    file7 = 'myyss.pkl'
    file8 = 'myyss2.pkl'
    files = [file1,file2,file3,file4,file5,file6,file7,file8]
    myvars = [myendingMSEm1a,myendingMSEmua,y,y2,ym,ym2,ys,ys2 ]
    for i in range(len(files)):
        f = open(files[i],'wb')
        pickle.dump(myvars[i],f)
        f.close()
        print(files[i]+'data saved')
    
    xlabels = ['MI regularization ('+r'$\lambda_2$'+')', 'Muscle Regularization ('+r'$\lambda_3$'+')']
    ylabels = ['Validation RMSE','Validation RMSE', 'Stable Pronated Instances', 'Stable Pronated Instances', 'Stable Midrange Instances', 'Stable Midrange Instances', 'Stable Supinated Instances', 'Stable Supinated Instances']
    loadedmeans=[]
    loadedmins= []
    loadedmaxs=[]
    loadedstd=[]
    #myloadedvars = [loadedmyendingMSEm1a,loadedmyendingMSEmua,loadedy,loadedy2,loadedym,loadedym2,loadedys,loadedys2 ]
    myloadedvars = []
    saveimagename = ['MIRMSEanalysis','MURMSEanalysis', 'MIStablepronated','MUStablepronated','MIStablemidrange','MUStablemidrange','MIStablesupinated','MUStablesupinated']
    for i in range(len(files)):
        f = open(files[i],'rb')
        myloadedvars.append(pickle.load(f))
        f.close()
    myloadedvars[0] = np.sqrt(myloadedvars[0])
    myloadedvars[1] = np.sqrt(myloadedvars[1])
    t,p = stats.ttest_ind(myloadedvars[2],myloadedvars[4])#comparing MI stable pronated and midrange
    t,p = stats.ttest_ind(myloadedvars[2],myloadedvars[6])#comparing MI stable pronated and supinated
    t,p = stats.ttest_ind(myloadedvars[6],myloadedvars[4])#comparing MI stable supinated and midrange
    t,p = stats.ttest_ind(myloadedvars[5],myloadedvars[3])#comparing MU stable pronated and midrange
    t,p = stats.ttest_ind(myloadedvars[5],myloadedvars[7])#comparing MU stable pronated and supinated
    t,p = stats.ttest_ind(myloadedvars[3],myloadedvars[7])#comparing MU stable pronated and supinated
    for i in range(len(files)):
        print(files[i]+' is '+str(np.mean(myloadedvars[i],axis=0)))
        loadedmeans.append(np.mean(myloadedvars[i],axis=0))
        loadedmins.append(np.amin(myloadedvars[i],axis=0))
        loadedmaxs.append(np.amax(myloadedvars[i],axis=0))
        loadedstd.append(np.std(myloadedvars[i],axis=0))
    for i in range(len(files)):
        plt.errorbar(m1as,loadedmeans[i],yerr=loadedstd[i])
        if (i == 0 or i==2 or i==4 or i==6):
            plt.xlabel(xlabels[0])
        else:
            plt.xlabel(xlabels[1])
        if (i !=0 and i!=1):
            plt.ylim([0,150])
        plt.ylabel(ylabels[i])
        plt.savefig('./Thesis/ThesisChapter/ThesisImg/'+saveimagename[i]+'.pdf')
        plt.close()
    for i in range(len(files)):
        plt.errorbar(m1as,loadedmeans[i],yerr=loadedstd[i])
        if (i == 0 or i==2 or i==4 or i==6):
            plt.xlabel(xlabels[0])
        else:
            plt.xlabel(xlabels[1])
        plt.ylabel(ylabels[i])
        plt.figure()
def neuronregplots(timesteps,epo,examples,nangle,kernm1a,kernmua,m1a,mua,patience):   
    neuronvec = [25,50,75,100,125]
    numruns=20
    myendingMSE = np.zeros([len(neuronvec),numruns])
    for i in range(len(neuronvec)):
        for j in range(numruns):
            runindex=j
            expectedoutput2,scores, mymodel, mymodel2,v,handmat, myweights, history,vslice,goat,output2,theta,pcount,mcount,scount,orientation,timer,theta2,vslicereshapepro,reshapedhandmatpro,proheatinput,proheatoutput,proforce,vslicereshapemid,reshapedhandmatmid,midheatinput,midheatoutput,midforce,vslicereshapesup,reshapedhandmatsup,supheatinput,supheatoutput,supforce,holdperiod=nanglesposcirlce(kernm1a,kernmua,patience,timesteps,epo,examples,neuronvec[i],m1a,mua,nangle,runindex)
            myendingMSE[i][j]=history['val_mean_squared_error'][-patience - 1]
            #print(myendingMSE[i][j])
            gc.collect()
            backend.clear_session()
    neuronfile='neuronMSEfile.pkl'
    f = open(neuronfile,'wb')
    pickle.dump(myendingMSE,f)
    f.close()
    print('data saved')
    
    
    loadedneurondata = []
    #loadedneuronmins = []
    #loadedneuronmaxs = []
    #loadedneuronmeans = []
    f = open(neuronfile,'rb')
    loadedneurondata.append(pickle.load(f))
    f.close()
    sqrtloadedneurondata = np.sqrt(loadedneurondata)
    loadedneuronmeans = np.mean(sqrtloadedneurondata[0],axis=1)
    loadedneuronmins = np.amin(sqrtloadedneurondata[0],axis=1)
    loadedneuronmaxs = np.amax(sqrtloadedneurondata[0],axis=1)
    loadedneuronstd = np.std(sqrtloadedneurondata[0],axis=1)
    t,p = stats.ttest_ind(loadedneurondata[0][3],loadedneurondata[0][4])
    plt.errorbar(neuronvec,loadedneuronmeans,yerr=loadedneuronstd)
    plt.xlabel("Neurons")
    plt.ylabel("Validation RMSE")
    plt.savefig('./Thesis/ThesisChapter/ThesisImg/neuronvsRMSEplot.pdf')
    