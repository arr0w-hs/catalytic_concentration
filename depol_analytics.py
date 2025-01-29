#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:34:53 2024

@author: hsharma4
for analysis of quantum states
"""

import qutip as qt
import sympy as sp
import numpy as np

zero = qt.basis(2,0)
one = qt.basis(2,1)
I = qt.qeye(2)
X = qt.sigmax()
Z = qt.sigmaz()
Y = qt.sigmay()
H = 1/np.sqrt(2)*(X+Z)


def new_state_pauli_x1(a_val, prob):
    """this error adds depolarizing noise to coherent error state"""

    phi_plus = sp.sqrt(0.5)*(qt.tensor(zero,zero) +
                             qt.tensor(one,one))

    phi_tilde = sp.sqrt(a_val)*phi_plus.full() + sp.sqrt((1-a_val)/3)*(
        (qt.tensor((Z), I)*phi_plus).full() + (qt.tensor((X), I)*phi_plus).full()+
        (qt.tensor((X*Z), I)*phi_plus).full())

    # phi_tilde = sp.sqrt(a_val)*phi_plus + sp.sqrt(1-a_val)*(
    #     qt.tensor(X, I)*phi_plus)

    #print(phi_tilde.dag()*phi_tilde)
    phi_tilde = qt.Qobj(phi_tilde, dims = [[2,2],[2,2]])

    bell_st1 = prob*qt.ket2dm(phi_tilde) + (1-prob)/3*(

        qt.ket2dm(qt.tensor(X,I)*phi_tilde)+
        qt.ket2dm(qt.tensor(Z,I)*phi_tilde)+
        qt.ket2dm(qt.tensor(X*Z,I)*phi_tilde)
        )

    #print(bell_st1)
    bell_st = qt.tensor(bell_st1, bell_st1)

    """putting ph in front of the four entangled spins"""
    bell_st = qt.tensor(qt.ket2dm(zero), bell_st)
    bell_st = bell_st.permute([0,1,3,2,4])
    assert np.real(bell_st.tr()) >= 0.9

    return bell_st



a_sym = sp.Symbol('a')
p_sym = sp.Symbol('p')

phi_plus = sp.sqrt(0.5)*(qt.tensor(zero,zero) +
                         qt.tensor(one,one))
print(phi_plus)
m = (X+Z)/2

m = qt.tensor(m,m)
print(m*phi_plus)
