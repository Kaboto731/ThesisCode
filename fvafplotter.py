#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 19:47:18 2020

@author: manuel
"""
import matplotlib.pyplot as plt
import numpy as np
def plotsavedfvafs():
    mymidx=[]
    mymidy=[]
    mysupx=[]
    mysupy=[]
    myprox=[]
    myproy=[]
    #midrange data
    i=0
    with open("midxfvaf.txt")  as midx:
        for line in midx:
            mymidx.append(float(line))
            i=i+1
    i=0
    with open("midyfvaf.txt")  as midx:
        for line in midx:
            mymidy.append(float(line))
            i=i+1
    #pronated data
    i=0
    with open("proxfvaf.txt")  as midx:
        for line in midx:
            myprox.append(float(line))
            i=i+1
    i=0
    with open("proyfvaf.txt")  as midx:
        for line in midx:
            myproy.append(float(line))
            i=i+1
    #supinated data
    i=0
    with open("supxfvaf.txt")  as midx:
        for line in midx:
            mysupx.append(float(line))
            i=i+1
    i=0
    with open("supyfvaf.txt")  as midx:
        for line in midx:
            mysupy.append(float(line))
            i=i+1
    #plotting
   
    x=np.linspace(0.5,1.05,5)
    y=x

    
    plt.scatter(mysupx,mysupy,c="g",s=10)
    plt.scatter(mymidx,mymidy,c="r",s=10)
    plt.scatter(myprox,myproy,c="b",s=10)
    plt.plot(x,y)
    plt.gca().set_xlim([0.8,1.005])
    plt.gca().set_ylim([0.8,1.005])
    plt.gca().set_aspect('equal')
    plt.legend(("y=x reference","supinated","midrange","pronated" ))
    #plt.legend(("supinated","midrange","pronated","y=x reference" ))
    plt.xlabel(("X Force FVAF"))
    plt.ylabel(("Y Force FVAF"))
    plt.xticks(np.arange(0.8,1.05, step=0.05))
    plt.yticks(np.arange(0.8,1.05 ,step=0.05))
    plt.savefig('myfvafcor.pdf',bbox_inches='tight')
    plt.show()
    print('variables:',len(mysupx))