#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:30:53 2024

@author: hsharma4
functions for creating depol channels
and Bell states going through them
"""
#import sys
#import os
#sys.path.append(os.path.dirname(__file__))
import numpy as np
import qutip as qt


zero = qt.basis(2,0)
one = qt.basis(2,1)
I = qt.qeye(2)
X = qt.sigmax()
Z = qt.sigmaz()
Y = qt.sigmay()
H = 1/np.sqrt(2)*(X+Z)

def r_state(alpha, prob):
    """func for r state"""

    bell1 = (prob*qt.ket2dm(np.sqrt(alpha)*qt.tensor(zero,zero) +
                    np.sqrt(1-alpha)*qt.tensor(one,one)) +
                 (1-prob)*qt.ket2dm(qt.tensor(zero, one)))
    bell_st = qt.tensor(bell1, bell1)

    """putting ph in front of the four entangled spins"""
    bell_st = qt.tensor(qt.ket2dm(zero), bell_st)
    bell_st = bell_st.permute([0,1,3,2,4])
    assert bell_st.tr() >= 0.9

    return bell_st

def s_state(alpha, prob):
    """func for s state"""

    bell1 = (prob*qt.ket2dm(np.sqrt(alpha)*qt.tensor(zero,zero) +
                    np.sqrt(1-alpha)*qt.tensor(one,one)) +
                 ((1-prob))*qt.ket2dm(qt.tensor(zero, zero)))
    bell_st = qt.tensor(bell1, bell1)

    """putting ph in front of the four entangled spins"""
    bell_st = qt.tensor(qt.ket2dm(zero), bell_st)
    bell_st = bell_st.permute([0,1,3,2,4])
    assert bell_st.tr() >= 0.9

    return bell_st

def werner_state(alpha, prob):
    """func for werner state"""

    bell1 = (prob*qt.ket2dm(np.sqrt(alpha)*qt.tensor(zero,zero) +
                    np.sqrt(1-alpha)*qt.tensor(one,one)) +
                 (1-prob)/4*(qt.tensor(I, I)))
    bell_st = qt.tensor(bell1, bell1)

    """putting ph in front of the four entangled spins"""
    bell_st = qt.tensor(qt.ket2dm(zero), bell_st)
    bell_st = bell_st.permute([0,1,3,2,4])
    assert bell_st.tr() >= 0.9

    return bell_st

def new_state_pauli_x1(a_val, prob):
    """this error adds depolarizing noise to coherent error state"""

    phi_plus = np.sqrt(0.5)*(qt.tensor(zero,zero) +
                             qt.tensor(one,one))

    phi_tilde = np.sqrt(a_val)*phi_plus + np.sqrt((1-a_val)/3)*(
        qt.tensor((Z), I)*phi_plus + qt.tensor((X), I)*phi_plus+
        qt.tensor((X*Z), I)*phi_plus)

    # phi_tilde = np.sqrt(a_val)*phi_plus + np.sqrt(1-a_val)*(
    #     qt.tensor(X, I)*phi_plus)

    #print(phi_tilde.dag()*phi_tilde)

    bell_st1 = prob*qt.ket2dm(phi_tilde) + (1-prob)/3*(
        #qt.ket2dm(phi_tilde)+
        qt.ket2dm(qt.tensor(X,I)*phi_tilde)+
        qt.ket2dm(qt.tensor(Z,I)*phi_tilde)+
        qt.ket2dm(qt.tensor(X*Z,I)*phi_tilde)
        )

    bell_st = qt.tensor(bell_st1, bell_st1)

    """putting ph in front of the four entangled spins"""
    bell_st = qt.tensor(qt.ket2dm(zero), bell_st)
    bell_st = bell_st.permute([0,1,3,2,4])
    assert np.real(bell_st.tr()) >= 0.9

    return bell_st

def new_state_pauli_x2(a_val, prob):
    """this error adds depolarizing noise to coherent error state"""

    phi_plus = np.sqrt(0.5)*(qt.tensor(zero,zero) +
                             qt.tensor(one,one))

    phi_tilde = np.sqrt(a_val)*phi_plus + np.sqrt((1-a_val)/3)*(
        qt.tensor((Z), I)*phi_plus + qt.tensor((X), I)*phi_plus+
        qt.tensor((X*Z), I)*phi_plus)


    bell_st1 = qt.ket2dm(phi_tilde)
    II = qt.tensor(I,I)

    bell_st = (qt.tensor(bell_st1, II)+qt.tensor(II, bell_st1))


    """putting ph in front of the four entangled spins"""
    bell_st = qt.tensor(qt.ket2dm(zero), bell_st)
    bell_st = bell_st.permute([0,1,3,2,4])
    # assert np.real(bell_st.tr()) >= 0.9

    return bell_st
