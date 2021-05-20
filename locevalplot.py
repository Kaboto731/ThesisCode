#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 12:26:23 2020

@author: manuel
"""

import numpy as np
import matplotlib.pyplot as plt

def locationplot(theta2):
    plt.rcParams.update({'font.size': 8})
    plt.axes(projection = "polar")
    
    
    for rad in theta2:
        plt.polar(rad,1,'o',color='b')
    plt.savefig('./Thesis/ThesisChapter/ThesisImg/locationplot.pdf')    
    plt.show()