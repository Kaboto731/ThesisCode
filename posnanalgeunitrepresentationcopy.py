#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 13:49:53 2020

@author: manuel
"""
import numpy as np
import math as ma
import random
from neuralnetv2 import buildneuralnet
from minjerk import min_jerk
import keras
#import h5py
from neuralnetv2 import myeslu
from neuralnetv2 import ESLU
from keras.layers import Activation
from numpy.random import seed
seed(42)# keras seed fixing
import tensorflow as tf
tf.random.set_random_seed(42)# tensorflow seed fixing
# from keras.callbacks import CSVLogger
import json
def nanglesposcirlce(kernm1a,kernmua,patience,timesteps,epo,examples,numRNNneurons,m1a,mua,nangles,runindex):

    #drop rate of neural network
    dropped=0.5
    #batch size
    batcher=512
    #number of neurons in RNN
    
    #Build of main program---------------------------------------
    #Alpha for the signal noise, a constant that is the coefficient of the noise
    #using page 38 we get that alpha was chosen to be 0.1
    alpha = 0.1
    #mu, which is the mean that our noise gauss is centered around
    mu = 0
    
    
    #Rho for defining posture:
    onehotmid=np.array([0,0,1])
    onehotpro=np.array([1,0,0])
    onehotsup=np.array([0,1,0])
    #our arm muscles
    MidOrientationdeg = np.array([159.4, 17.9, 1.2, 341.4, 313.9, 197.8])
    ProneOrientationdeg = np.array([89.4, 355.5, 310.7, 280.5, 263, 179.1])
    SupinatedOrientationdeg = np.array([218.6, 69.6, 42.4, 8.1, 321.8, 254.3])
    #converting to radians
    MidOrientation = MidOrientationdeg*ma.pi/180
    ProneOrientation =ProneOrientationdeg*ma.pi/180
    SupinatedOrientation = SupinatedOrientationdeg*ma.pi/180   
    
    matmid = np.transpose(np.array([np.cos(MidOrientation),np.sin(MidOrientation)]))
    matpro =   np.transpose(np.array([np.cos(ProneOrientation),np.sin(ProneOrientation)]))
    matsup =   np.transpose(np.array([np.cos(SupinatedOrientation),np.sin(SupinatedOrientation)]))
    
    #visual neurons for the test, using page 38 we have 107 neurons with an average 
    #activation of 1
    
    #Start of program:
    #PD = np.random.uniform(-ma.pi,ma.pi,n)
    #from page 25 we get PD's are evenly distributed around the unit circle from 
    #(-pi to pi), we have n cells
    PD = np.linspace(-ma.pi,ma.pi,numRNNneurons)
    #taking the x and y components of these cells
    PDx = np.cos(PD)
    PDy = np.sin(PD)
    #putting them in an array
    PDvec=np.array([PDx,PDy])
    #our noise standard deviation of the signal       
    noisesigma = np. zeros([examples,numRNNneurons])
    #our actual added noise
    noise= np.zeros([examples,numRNNneurons])  
    #the timer for the monkey, with  0.05 timesteps
    #timer = np.arange(0,5.5,0.5)
    timer = np.linspace(0,5,timesteps)
    holdperiod =0 # we hold for half a second
    for i in range(len(timer)):
        if (timer[i]<0.5):
            holdperiod = holdperiod+1
            
    traingosignal = np.zeros([examples,timesteps])
    #an array holding the index of the go signal for each example
    goat = np.zeros(examples)
    vslice = np.zeros([examples,timesteps,numRNNneurons+4])
    theta = np.random.uniform(0,2*np.pi,examples)-np.pi
    thetax = np.zeros(examples)
    thetay = np.zeros(examples)
    #the wrist posture for each example
    orientation = np.zeros(examples)
    thetavec = np.zeros([2,examples])
    v = np.zeros([examples,numRNNneurons])
    myinner= np.zeros([examples,numRNNneurons])
    jerk = np.zeros([int(examples),2,2])
    output2 = np.zeros([examples,len(timer),2])
    handmat = np.zeros([examples,6,2])
    #count of wrist postures
    pcount=0
    mcount=0
    scount=0
    thetax= np.cos(theta)
    thetay = np.sin(theta)
    #making a vector
    thetavec= np.array([thetax,thetay])
    for i in range(examples):
    
    #chosing a random posture
        orientation[i] = np.random.randint(1,4)
    
    
    #taking x/y elements on random target degree
    #doing the visual input function
        for z in range(numRNNneurons):
            myinner[i,z] = np.matmul(np.transpose(PDvec[:,z]),thetavec[:,i])
            myexp=-(myinner[i,z])
            v[i,z]= np.exp(myexp)
    #making the onehot encoding appended
        if (orientation[i] ==3):
                vslice[i,:,0:numRNNneurons+3]= np.append(v[i],onehotpro)
                handmat[i] = np.array(matpro)
                pcount=pcount+1
        if (orientation[i] ==2):
               vslice[i,:,0:numRNNneurons+3]=np.append(v[i],onehotmid)
               handmat[i] = np.array(matmid)
               mcount=mcount+1
        if (orientation[i] == 1):
                vslice[i,:,0:numRNNneurons+3] = np.append(v[i],onehotsup)
                handmat[i]=np.array(matsup)
                scount= scount+1
        noisesigma[i,:] = alpha*vslice[i,0,0:numRNNneurons] 
        for j in range(0,numRNNneurons):
                
                #the variance is decided by alpha of our noise, thus stronger visual inputs
                #have greater variance
                noise[i,j] = random.gauss(mu,noisesigma[i,j])
                vslice[i,:,j]= vslice[i,:,j]+noise[i,j]
    
    #our fixed go signal starting
        goat[i]=random.randint(20,40) 
     #   goat2[i] = random.randint(1,2)
        for j in range(len(timer)):
            #monkey was required to hold position 0.5 seconds after go signal was presented
                if (j>=goat[i]):
                    traingosignal[i,j]=1 
        vslice[i,:,numRNNneurons+3]=traingosignal[i]
        jerk[i,1,:] = thetavec[:,i]
        jerkmotion2 =np.zeros((len(timer)+1-int(goat[i]),2))
        #we hold for a half second
        jerkmotion2 = min_jerk(jerk[i],len(timer)+1-goat[i]-holdperiod,[],[],[])
        for j in range(int(goat[i]),len(timer)):
            if (j<len(timer)-holdperiod):
                output2[i][j]=jerkmotion2[j+1-int(goat[i])]
            else:
                output2[i][j]=output2[i][len(timer)-holdperiod-1]
    #If we do not load the nn file we have to train a new one
    neuronstr = 'neurons'+str(numRNNneurons)
    if (runindex ==0):
        runindexstr = ''
    else:
        runindexstr = 'r'+str(runindex)
    try: 
        kernm1astr ='kernm1a'+str(int(1/kernm1a))
    except:
        kernm1astr = 'kernm1astr0'
    try: 
        kernmuastr ='kernmua'+str(int(1/kernmua))
    except:
        kernmuastr = 'kernmuastr0'
    try:
        m1astr='m1a'+str(int(1/m1a))
    except:
        m1astr = 'm1a0'
    try:
        muastr='mua'+str(int(1/mua))
    except:
        muastr = 'mua0'
    [mymodel,mymodel2]= buildneuralnet(timesteps,numRNNneurons+4,dropped,numRNNneurons,m1a,mua,kernm1a,kernmua)
    try:
        mymodel = keras.models.load_model('model'+neuronstr+m1astr+kernm1astr+muastr+kernmuastr+runindexstr, custom_objects={'myeslu': myeslu})
        mymodel2 = keras.models.load_model('model2'+neuronstr+m1astr+kernm1astr+muastr+kernmuastr+runindexstr, custom_objects={'myeslu': myeslu})
        
        #mymodel.load_weights('weights'+neuronstr+m1astr+kernm1astr+muastr+kernmuastr+runindexstr+'.h5')
    except:
    
        #the outer layer and inner layer of the neural net, inner being mymodel2, and outer being mymodel
        #csv_logger = CSVLogger('training.log', separator=',', append=False)
        mymodel.compile(optimizer='adam',loss='mean_squared_error',metrics=['mse'])#loss='mean_squared_error',metrics=['mse']
        mc = keras.callbacks.ModelCheckpoint('model'+neuronstr+m1astr+kernm1astr+muastr+kernmuastr+runindexstr, monitor='val_loss', mode='min', verbose=1,save_best_only=True)
        mycallbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=patience,verbose=0, mode='auto'),mc]
        history = mymodel.fit(x=[vslice,handmat],y=[output2],epochs=epo,batch_size=batcher,verbose=1,validation_split=0.2,callbacks = mycallbacks)#validation_split=0.2,batch_size=batcher
        history_dict = history.history
        json.dump(history_dict, open('myhist'+neuronstr+m1astr+kernm1astr+muastr+kernmuastr+runindexstr+'.json', 'w'))
        mymodel.save_weights('weights'+neuronstr+m1astr+kernm1astr+muastr+kernmuastr+runindexstr+'.h5')
        mymodel2.save('model2'+neuronstr+m1astr+kernm1astr+muastr+kernmuastr+runindexstr)
        mymodel = keras.models.load_model('model'+neuronstr+m1astr+kernm1astr+muastr+kernmuastr+runindexstr, custom_objects={'myeslu': Activation(myeslu)})   
    history = json.load(open('myhist'+neuronstr+m1astr+kernm1astr+muastr+kernmuastr+runindexstr+'.json', 'r'))
    mymodel.summary()
    mymodel2.summary()
    scores = mymodel.evaluate(x=[vslice,handmat],y=[output2], verbose=0)
    #after training nangles directions with the same go signal we want to get a slice of the different degrees
    #number of postures * target locations'
    # onehotmid=np.array([0,0,1])
    # onehotpro=np.array([1,0,0])
    # onehotsup=np.array([0,1,0])
    # PD = np.linspace(-ma.pi,ma.pi,numRNNneurons)
    # #taking the x and y components of these cells
    # PDx = np.cos(PD)
    # PDy = np.sin(PD)
    # #putting them in an array
    # PDvec=np.array([PDx,PDy])
    # nangles = 16
    # MidOrientationdeg = np.array([159.4, 17.9, 1.2, 341.4, 313.9, 197.8])
    # ProneOrientationdeg = np.array([89.4, 355.5, 310.7, 280.5, 263, 179.1])
    # SupinatedOrientationdeg = np.array([218.6, 69.6, 42.4, 8.1, 321.8, 254.3])
    # #converting to radians
    # MidOrientation = MidOrientationdeg*ma.pi/180
    # ProneOrientation =ProneOrientationdeg*ma.pi/180
    # SupinatedOrientation = SupinatedOrientationdeg*ma.pi/180  
    # mu=0
    # alpha = 0.1
    m=6
 
    
    matmid = np.transpose(np.array([np.cos(MidOrientation),np.sin(MidOrientation)]))
    matpro =   np.transpose(np.array([np.cos(ProneOrientation),np.sin(ProneOrientation)]))
    matsup =   np.transpose(np.array([np.cos(SupinatedOrientation),np.sin(SupinatedOrientation)]))
    proheatinput = np.zeros([nangles,timesteps,numRNNneurons])
    midheatinput = np.zeros([nangles,timesteps,numRNNneurons])
    supheatinput = np.zeros([nangles,timesteps,numRNNneurons])
    proheatoutput = np.zeros([nangles,timesteps,m])
    midheatoutput = np.zeros([nangles,timesteps,m])
    supheatoutput = np.zeros([nangles,timesteps,m])
    proforce= np.zeros([nangles,timesteps,2])
    midforce= np.zeros([nangles,timesteps,2])
    supforce= np.zeros([nangles,timesteps,2])
    theta2x = np.zeros(nangles)
    theta2y = np.zeros(nangles)
    theta2= np.zeros(nangles)
    v2 = np.zeros([nangles,numRNNneurons])
    handmatsup = np.zeros([6,2])
    handmatmid = np.zeros([6,2])
    handmatpro = np.zeros([6,2])
    theta2vec = np.zeros([2,nangles])
    vslicesup = np.zeros([nangles,timesteps,numRNNneurons+4])
    vslicemid = np.zeros([nangles,timesteps,numRNNneurons+4])
    vslicepro = np.zeros([nangles,timesteps,numRNNneurons+4])
    noisesigma2 = np.zeros([nangles,numRNNneurons])
    noise2 = np.zeros([nangles,numRNNneurons])
    myinner2= np.zeros([nangles,numRNNneurons])
    jerk2 = np.zeros([int(nangles),2,2])
    expectedoutput2 = np.zeros([nangles,timesteps,2])
    for i in range(nangles):
        theta2[i] = (i*(2*ma.pi/nangles))%(2*ma.pi)-ma.pi
        #chosing a fixed posture

        #taking x/y elements on random target degree
    theta2x= np.cos(theta2)
    theta2y = np.sin(theta2)
    #making a vector
    theta2vec= np.array([theta2x,theta2y])
    #doing the visual input function
    for i in range(nangles):
        for z in range(numRNNneurons):
            myinner2[i,z] = np.matmul(np.transpose(PDvec[:,z]),theta2vec[:,i])
            myexp=-(myinner2[i,z])
            v2[i,z]= np.exp(myexp)
    #making the onehot encoding appended
        vslicepro[i,:,0:numRNNneurons+3]= np.append(v2[i],onehotpro)
        handmatpro = np.array(matpro)
        vslicemid[i,:,0:numRNNneurons+3]=np.append(v2[i],onehotmid)
        handmatmid = np.array(matmid)
        vslicesup[i,:,0:numRNNneurons+3] = np.append(v2[i],onehotsup)
        handmatsup=np.array(matsup)
        noisesigma2[i,:] = alpha*vslicesup[i,0,0:numRNNneurons] 
        for j in range(0,numRNNneurons):
                
                #the variance is decided by alpha of our noise, thus stronger visual inputs
                #have greater variance
                noise2[i,j] = random.gauss(mu,noisesigma2[i,j])
                vslicesup[i,:,j]= vslicesup[i,:,j]+noise2[i,j]
                vslicemid[i,:,j]= vslicemid[i,:,j]+noise2[i,j]
                vslicepro[i,:,j]= vslicepro[i,:,j]+noise2[i,j]
        #our fixed go signal starting
        goat[i]=30  
        traingosignal2=np.zeros([nangles,timesteps])
     #   goat2[i] = random.randint(1,2)
        for j in range(timesteps):
            #monkey was required to hold position 0.5 seconds after go signal was presented
                if (j>=goat[i]):
                    traingosignal2[i,j]=1 
        jerk2[i,1,:] = theta2vec[:,i]
        jerkmotion2 =np.zeros((timesteps+1-int(goat[i]),2))
        #we hold for a half second
        jerkmotion2 = min_jerk(jerk2[i],timesteps+1-goat[i]-holdperiod,[],[],[])
        for j in range(int(goat[i]),timesteps):
            if (j<timesteps-holdperiod):
                expectedoutput2[i][j]=jerkmotion2[j+1-int(goat[i])]
            else:
                expectedoutput2[i][j]=expectedoutput2[i][len(timer)-holdperiod-1]
        
        vslicesup[i,:,numRNNneurons+3]=traingosignal2[i]
        vslicemid[i,:,numRNNneurons+3]=traingosignal2[i]
        vslicepro[i,:,numRNNneurons+3]=traingosignal2[i]
            #predict for nangles
        vslicereshapepro = np.reshape(vslicepro[i],(1,timesteps,numRNNneurons+4))
        reshapedhandmatpro=np.reshape(handmatpro,(1,6,2))
        myoutput= mymodel.predict(x=[vslicereshapepro,reshapedhandmatpro])
        myoutput2= mymodel2.predict(x=[vslicereshapepro,reshapedhandmatpro])
        proheatinput[i] = myoutput2[1]
        proheatoutput[i] = myoutput2[2]
        proforce[i] = myoutput
        vslicereshapemid = np.reshape(vslicemid[i],(1,timesteps,numRNNneurons+4))
        reshapedhandmatmid=np.reshape(handmatmid,(1,6,2))
        myoutput= mymodel.predict(x=[vslicereshapemid,reshapedhandmatmid])
        myoutput2= mymodel2.predict(x=[vslicereshapemid,reshapedhandmatmid])
        midheatinput[i] = myoutput2[1]
        midheatoutput[i]=myoutput2[2]
        midforce[i] = myoutput
        #
        vslicereshapesup = np.reshape(vslicesup[i],(1,timesteps,numRNNneurons+4))
        reshapedhandmatsup=np.reshape(handmatsup,(1,6,2))
        myoutput= mymodel.predict(x=[vslicereshapesup,reshapedhandmatsup])
        myoutput2= mymodel2.predict(x=[vslicereshapesup,reshapedhandmatsup])
        supheatinput[i] = myoutput2[1]
        supheatoutput[i]=myoutput2[2]
        supforce[i]=myoutput
    myweights = mymodel.get_weights()   
        
    return expectedoutput2,scores, mymodel, mymodel2,v,handmat, myweights, history,vslice,goat,output2,theta,pcount,mcount,scount,orientation,timer,theta2,vslicereshapepro,reshapedhandmatpro,proheatinput,proheatoutput,proforce,vslicereshapemid,reshapedhandmatmid,midheatinput,midheatoutput,midforce,vslicereshapesup,reshapedhandmatsup,supheatinput,supheatoutput,supforce,holdperiod
        
        