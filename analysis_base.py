#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 30 10:34:56 2023

@author: hsharma4
"""


import numpy as np
import scipy as sc
import math as math

import sys
import os
sys.path.append(os.path.abspath("//Users/hsharma4/Desktop/Multipartite state concentration/GHZ state project/robustness"))
from locc_base import *
from slocc_base import *
from state_change import *

def fwhm_calculator(fidelity_list):
    list_of_higher_args = []
    width = 0
    max_arg = np.argmax(fidelity_list)
    max_val = np.max(fidelity_list)
    min_val = np.min(fidelity_list)
    #print(len(fidelity_list))
    for i in range(len(fidelity_list)):
        if fidelity_list[i] >= (0.95*max_val):#+0.05*min_val):
            list_of_higher_args.append(i)
    #print(np.max(list_of_higher_args) , np.min(list_of_higher_args))
    #width = np.max(list_of_higher_args) - np.min(list_of_higher_args)
    width = (np.max(list_of_higher_args) - np.min(list_of_higher_args))/len(fidelity_list)
    
    return width

def right_width(fidelity_list):
    list_of_higher_args = []
    width = 0
    max_arg = np.argmax(fidelity_list)
    max_val = np.max(fidelity_list)
    min_val = np.min(fidelity_list)
    for i in range(len(fidelity_list)):
        if fidelity_list[i] >= (0.95*max_val):#+0.05*min_val):
            list_of_higher_args.append(i)
    
    width = np.max(list_of_higher_args) - max_arg
    
    return width

def left_width(fidelity_list):
    list_of_higher_args = []
    width = 0
    max_arg = np.argmax(fidelity_list)
    max_val = np.max(fidelity_list)
    min_val = np.min(fidelity_list)
    for i in range(len(fidelity_list)):
        if fidelity_list[i] >= (0.95*max_val):#+0.05*min_val):
            list_of_higher_args.append(i)
    
    width = max_arg - np.min(list_of_higher_args)
    
    return width

def double_derivative(fidelity_list, y):
    #der = np.gradient(fidelity_list)
    max_arg = np.argmax(fidelity_list)
    if max_arg == 0:
        double_der = (fidelity_list[max_arg+1]+fidelity_list[max_arg+1]-2*fidelity_list[max_arg])/(y[max_arg+1]-y[max_arg])**2
    elif max_arg == len(fidelity_list)-1:
        double_der = (fidelity_list[max_arg-1]+fidelity_list[max_arg-1]-2*fidelity_list[max_arg])/(y[max_arg]-y[max_arg-1])**2
    else:
        double_der = (fidelity_list[max_arg+1]+fidelity_list[max_arg-1]-2*fidelity_list[max_arg])/(y[max_arg+1]-y[max_arg])**2
        
    if -1/double_der >= 10:
        print(fidelity_list)
    return double_der#np.gradient(der)