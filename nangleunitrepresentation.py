#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 20 12:38:30 2020

@author: manuel

"""
import numpy as np
import math as ma
import random
import keras
from minjerk import min_jerk
from neuralnetv2 import buildneuralnet

def nanglescircle(timesteps,epo,examples,numRNNneurons,savennfile,loadnnfile,m1a,mua,nangles):

            #drop rate of neural network
            dropped=0.5
            #batch size
            batcher=512
            #number of neurons in RNN
            numRNNneurons=50
            
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
            n = numRNNneurons
            
            #Start of program:
            #PD = np.random.uniform(-ma.pi,ma.pi,n)
            #from page 25 we get PD's are evenly distributed around the unit circle from 
            #(-pi to pi), we have n cells
            PD = np.linspace(-ma.pi,ma.pi,n)
            #taking the x and y components of these cells
            PDx = np.cos(PD)
            PDy = np.sin(PD)
            #putting them in an array
            PDvec=np.array([PDx,PDy])
            #our noise standard deviation of the signal       
            noisesigma = np.zeros([examples,n])
            #our actual added noise
            noise= np.zeros([examples,n])  
            #the timer for the monkey, with  0.05 timesteps
            #timer = np.arange(0,5.5,0.5)
            timer = np.linspace(0,5,timesteps)
            holdperiod =0 # we hold for half a second
            for i in range(len(timer)):
                if (timer[i]<0.5):
                    holdperiod = holdperiod+1
                    
            traingosignal = np.zeros([examples,len(timer)])
            #an array holding the index of the go signal for each example
            goat = np.zeros(examples)
            vslice = np.zeros([examples,len(timer),n+4])
            theta = np.random.uniform(0,15,examples)
            theta = theta*(2*np.pi/nangles)-np.pi
            thetax = np.zeros(examples)
            thetay = np.zeros(examples)
            #the wrist posture for each example
            orientation = np.zeros(examples)
            thetavec = np.zeros([2,examples])
            v = np.zeros([examples,n])
            myinner= np.zeros([examples,n])
            jerk = np.zeros((int(examples),2,2))
            output2 = np.zeros((examples,len(timer),2))
            handmat = np.zeros((examples,6,2))
            #count of wrist postures
            pcount=0
            mcount=0
            scount=0
            thetax = np.cos(theta)
            thetay = np.cos(theta)
            thetavec= np.array([thetax,thetay])
            for i in range(examples):
            
            #chosing a random posture
                orientation[i] = np.random.randint(1,4)
           
            #doing the visual input function
                for z in range(n):
                    myinner[i,z] = np.matmul(np.transpose(PDvec[:,z]),thetavec[:,i])
                    myexp=-(myinner[i,z])
                    v[i,z]= np.exp(myexp)
            #making the onehot encoding appended
                if (orientation[i] ==3):
                        vslice[i,:,0:n+3]= np.append(v[i],onehotpro)
                        handmat[i] = np.array(matpro)
                        pcount=pcount+1
                if (orientation[i] ==2):
                       vslice[i,:,0:n+3]=np.append(v[i],onehotmid)
                       handmat[i] = np.array(matmid)
                       mcount=mcount+1
                if (orientation[i] == 1):
                        vslice[i,:,0:n+3] = np.append(v[i],onehotsup)
                        handmat[i]=np.array(matsup)
                        scount= scount+1
                noisesigma[i,:] = alpha*vslice[i,0,0:n] 
                for j in range(0,n):
                        
                        #the variance is decided by alpha of our noise, thus stronger visual inputs
                        #have greater variance
                        noise[i,j] = random.gauss(mu,noisesigma[i,j])
                        vslice[i,:,j]= vslice[i,:,j]+noise[i,j]
            
            #our fixed go signal starting
                goat[i]=30  
             #   goat2[i] = random.randint(1,2)
                for j in range(len(timer)):
                    #monkey was required to hold position 0.5 seconds after go signal was presented
                        if (j>=goat[i]):
                            traingosignal[i,j]=1 
                vslice[i,:,n+3]=traingosignal[i]
                jerk[i,1,:] = thetavec[:,i]
                jerkmotion2 =np.zeros((len(timer)+1-int(goat[i]),2))
                #we hold for a half second
                jerkmotion2 = min_jerk(jerk[i],len(timer)+1-goat[i]-holdperiod,[],[],[])
                for j in range(int(goat[i]),len(timer)):
                    if (j<len(timer)-holdperiod):
                        output2[i][j]=jerkmotion2[j+1-int(goat[i])]
                    else:
                        output2[i][j]=output2[i][len(timer)-holdperiod-1]

            #the outer layer and inner layer of the neural net, inner being mymodel2, and outer being mymodel
            [mymodel,mymodel2]= buildneuralnet(len(timer),n+4,dropped,numRNNneurons,m1a,mua)
            mc = keras.callbacks.ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', verbose=1,save_best_only=True)
            mycallbacks = [keras.callbacks.EarlyStopping(monitor='val_loss',min_delta=0,patience=50,verbose=0, mode='auto'),mc]
            mymodel.compile(optimizer='adam',loss='mean_squared_error',metrics=['mse'])
            mymodel.summary()
            mymodel2.summary()
            history = mymodel.fit(x=[vslice,handmat],y=[output2],epochs=epo,batch_size=batcher,verbose=0,validation_split=0.2,callbacks = mycallbacks)
            mymodel = keras.models.load_model('best_model.h5')
            scores = mymodel.evaluate(x=[vslice,handmat],y=[output2], verbose=0)
            return scores
            
            