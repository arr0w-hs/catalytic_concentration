#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:30:53 2024

@author: hsharma4
functions for creating depol channels
and Bell states going through them
"""
import sys
import os
sys.path.append(os.path.dirname(__file__))
from base_locc import *
from base_slocc import *
from base_state_change import *
from base_siv_state_prep import *
from base_catalyst import *

import numpy as np
import scipy as sc
import pandas as pd
import matplotlib.pyplot as plt
from qutip import *
from qutip.measurement import measure, measurement_statistics, measure_observable

zero = basis(2,0)
one = basis(2,1)
I = qeye(2)
X = sigmax()
Z = sigmaz()
Y = sigmay()
H = 1/np.sqrt(2)*(X+Z)

def r_state(alpha, prob):

    bell1 = (prob*ket2dm(np.sqrt(alpha)*tensor(zero,zero) +
                    np.sqrt(1-alpha)*tensor(one,one)) +
                 (1-prob)*ket2dm(tensor(zero, one)))
    bell_st = tensor(bell1, bell1)
    
    """putting ph in front of the four entangled spins"""
    bell_st = tensor(ket2dm(zero), bell_st)
    bell_st = bell_st.permute([0,1,3,2,4])
    if bell_st.tr() < 0.9:
        raise Exception ("trace less than one")
        
    return bell_st

def s_state(alpha, prob):
    
    bell1 = (prob*ket2dm(np.sqrt(alpha)*tensor(zero,zero) +
                    np.sqrt(1-alpha)*tensor(one,one)) +
                 ((1-prob))*ket2dm(tensor(zero, zero)))
    bell_st = tensor(bell1, bell1)
    
    """putting ph in front of the four entangled spins"""
    bell_st = tensor(ket2dm(zero), bell_st)
    bell_st = bell_st.permute([0,1,3,2,4])
    if bell_st.tr() < 0.9:
        raise Exception ("trace less than one")
        
    return bell_st

def werner_state(alpha, prob):
    
    bell1 = (prob*ket2dm(np.sqrt(alpha)*tensor(zero,zero) +
                    np.sqrt(1-alpha)*tensor(one,one)) +
                 (1-prob)/4*(tensor(I, I)))
    bell_st = tensor(bell1, bell1)
    
    """putting ph in front of the four entangled spins"""
    bell_st = tensor(ket2dm(zero), bell_st)
    bell_st = bell_st.permute([0,1,3,2,4])
    if bell_st.tr() < 0.9:
        raise Exception ("trace less than one")
        
    return bell_st

def new_state_depol(a, prob):
    
    phi_plus = np.sqrt(0.5)*(tensor(zero,zero) +
                             tensor(one,one))
    
    phi_tilde = np.sqrt(a)*phi_plus + np.sqrt((1-a)/3)*tensor((X+Z+X*Z), I)*phi_plus
    
    bell_st1 = prob*ket2dm(phi_tilde) + (1-prob)*(
        a*ket2dm(phi_plus)+
        (1-a)/3*ket2dm(tensor(X,I)*phi_plus)+
        (1-a)/3*ket2dm(tensor(Z,I)*phi_plus)+
        (1-a)/3*ket2dm(tensor(X*Z,I)*phi_plus)
        )
    
    bell_st = tensor(bell_st1, bell_st1)
    
    """putting ph in front of the four entangled spins"""
    bell_st = tensor(ket2dm(zero), bell_st)
    bell_st = bell_st.permute([0,1,3,2,4])
    if np.real(bell_st.tr()) < 0.9:
        raise Exception ("trace less than one")
        
    return bell_st
    
def new_state_pauli_z(a, prob):
    
    phi_plus = np.sqrt(0.5)*(tensor(zero,zero) +
                             tensor(one,one))
    
    phi_tilde = np.sqrt(a)*phi_plus + np.sqrt((1-a))*tensor((Z), I)*phi_plus
    
    bell_st1 = prob*ket2dm(phi_tilde) + (1-prob)*(
        a*ket2dm(phi_plus)+
        (1-a)*ket2dm(tensor(Z,I)*phi_plus)#+
        #ket2dm(tensor(Z,I)*phi_plus)+
        #ket2dm(tensor(X*Z,I)*phi_plus)
        )
    #print(bell_st1)
    bell_st = tensor(bell_st1, bell_st1)
    
    """putting ph in front of the four entangled spins"""
    bell_st = tensor(ket2dm(zero), bell_st)
    bell_st = bell_st.permute([0,1,3,2,4])
    if np.real(bell_st.tr()) < 0.9:
        raise Exception ("trace less than one")
        
    return bell_st
    
def new_state_pauli_x(a, prob):
    
    phi_plus = np.sqrt(0.5)*(tensor(zero,zero) +
                             tensor(one,one))
    
    phi_tilde = np.sqrt(a)*phi_plus + np.sqrt((1-a))*tensor((X), I)*phi_plus
    
    bell_st1 = prob*ket2dm(phi_tilde) + (1-prob)*(
        a*ket2dm(phi_plus)+
        (1-a)*ket2dm(tensor(X,I)*phi_plus)#+
        #ket2dm(tensor(Z,I)*phi_plus)+
        #ket2dm(tensor(X*Z,I)*phi_plus)
        )
    #print(bell_st1)
    bell_st = tensor(bell_st1, bell_st1)
    
    """putting ph in front of the four entangled spins"""
    bell_st = tensor(ket2dm(zero), bell_st)
    bell_st = bell_st.permute([0,1,3,2,4])
    if np.real(bell_st.tr()) < 0.9:
        raise Exception ("trace less than one")
        
    return bell_st

def new_state_pauli_x1(a, prob):
    
    phi_plus = np.sqrt(0.5)*(tensor(zero,zero) +
                             tensor(one,one))
    # phi_tilde = np.sqrt(a)*phi_plus + np.sqrt((1-a)/3)*(
    #     tensor((Z), I)*phi_plus + tensor((X), I)*phi_plus+
    #     tensor((X*Z), I)*phi_plus)
    phi_tilde = np.sqrt(a)*phi_plus + np.sqrt((1-a))*tensor((Z), I)*phi_plus
    
    bell_st1 = prob*ket2dm(phi_tilde) + (1-prob)/3*(

        ket2dm(tensor(X,I)*phi_tilde)+
        ket2dm(tensor(Z,I)*phi_tilde)+
        ket2dm(tensor(X*Z,I)*phi_tilde)
        )

    #print(bell_st1)
    bell_st = tensor(bell_st1, bell_st1)
    
    """putting ph in front of the four entangled spins"""
    bell_st = tensor(ket2dm(zero), bell_st)
    bell_st = bell_st.permute([0,1,3,2,4])
    if np.real(bell_st.tr()) < 0.9:
        raise Exception ("trace less than one")
        
    return bell_st
    
