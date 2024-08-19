#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 10:51:19 2023

@author: hsharma4
"""

import numpy as np
import scipy as sc
import math as math
import pandas as pd
import matplotlib.pyplot as plt
from qutip import *
from qutip.measurement import measure, measurement_statistics, measure_observable

import sys
import os
sys.path.append(os.path.dirname(__file__))
from base_locc import *
from base_slocc import *
from base_state_change import *
from base_siv_state_prep import *
from base_catalyst import *
from updated_measurements import *
from syn2depol import depol_channel


Y = sigmay()
X = sigmax()
Z = sigmaz()
I = qeye(2)
H = 1/np.sqrt(2)*(X+Z)
ketbra0 = ket2dm(basis(2, 0))
ketbra1 = ket2dm(basis(2, 1))
plus = ket2dm((basis(2, 0)+basis(2, 1)).unit())
minus = ket2dm((basis(2, 0)-basis(2, 1)).unit())

S = ketbra0 + 1j*ketbra1


"""for a given permutation number from 0 to 5, returns the permutation for alice and bob"""
def permutation_unitary(permutation_num):
    if permutation_num >5:
        raise Exception ("wrong permutation number")
        
    perm_mat_alice = I
    perm_mat_bob = I
    for i in range(permutation_num):
        perm_mat_alice = ((i+1)%2)*(H*perm_mat_alice)+(i%2)*(perm_mat_alice*S*H)
        perm_mat_bob = ((i+1)%2)*(H*perm_mat_bob)+(i%2)*(perm_mat_bob*H*S*H*S)
        
    return perm_mat_alice, perm_mat_bob

"""if qubit is 0, bell state in spin is permuted
   if qubit is 1, bell state in nu is permuted"""
def permutation_distillation(psn, permutation_num, qubit):
    perm_alice, perm_bob = permutation_unitary(permutation_num)
    
    perm_mat_alice = (qubit%2)*tensor(I, perm_alice, I,I,I)+((qubit+1)%2)*tensor(I,I, perm_alice, I,I)
    perm_mat_bob = (qubit%2)*tensor(I,I,I, perm_bob, I)+((qubit+1)%2)*tensor(I,I,I,I, perm_bob)    
    
    psn = perm_mat_alice*perm_mat_bob*psn*perm_mat_alice.dag()*perm_mat_bob.dag()
    
    return psn
   
"""psn is the psn density matrix
   person is alice or bob
   basis is z (0), x (1) or y (2) bases
   target can be spin (0) or nu (1)"""

def coinc_distillation1(psn, mea_basis, control, err_rate):
    
    if control == 0:
        psn = psn
    elif control == 1:
        psn = swap_en_b(err_rate, err_rate)*psn*swap_en_b(err_rate, err_rate).dag()
    elif control ==2:
        psn = swap_en_a(err_rate, err_rate)*psn*swap_en_a(err_rate, err_rate).dag()
    else:
        psn = swap_en_a(err_rate, err_rate)*psn*swap_en_a(err_rate, err_rate).dag()
        psn = swap_en_b(err_rate, err_rate)*psn*swap_en_b(err_rate, err_rate).dag()
    
    zero = basis(2,0)
    one = basis(2,1)
    bell_st = 1/np.sqrt(2)*(tensor(zero,zero,zero,zero,zero) + 
                                     tensor(zero,one,zero,one,zero))
    
    if mea_basis == 1:
        psn = tensor(I,I,H,I,H)*psn*tensor(I,I,H,I,H).dag()
    elif mea_basis == 2:
        psn = tensor(I,I,S*H,I,S*H).dag()*psn*tensor(I,I,S*H,I,S*H)
    
    bell_st1 = tensor(I, Z, I, I, I)*bell_st
    prob_dist = bell_st.dag()*psn*bell_st + bell_st1.dag()*psn*bell_st1
    #print(prob_dist)
    fid_dist = (bell_st.dag()*psn*bell_st/prob_dist[0,0])[0,0]
    psn_final = fid_dist*bell_st*bell_st.dag() + bell_st1.dag()*psn*bell_st1*bell_st1*bell_st1.dag()

    return psn_final, np.real(fid_dist), np.real(prob_dist[0,0])

def coinc_distillation_final(psn, mea_basis, control, err_rate):
    
    if control == 0:
        psn = psn
    elif control == 1:
        psn = swap_en_b(err_rate, err_rate)*psn*swap_en_b(err_rate, err_rate).dag()
    #elif control ==2:
    #    psn = swap_en_a(err_rate, err_rate)*psn*swap_en_a(err_rate, err_rate).dag()
    #else:
    #    psn = swap_en_a(err_rate, err_rate)*psn*swap_en_a(err_rate, err_rate).dag()
    #    psn = swap_en_b(err_rate, err_rate)*psn*swap_en_b(err_rate, err_rate).dag()
    
    zero = basis(2,0)
    one = basis(2,1)
    bell_st = 1/np.sqrt(2)*(tensor(zero,zero,zero,zero,zero) + 
                                     tensor(zero,one,zero,one,zero))
    
    if mea_basis == 1:
        psn = tensor(I,I,H,I,H)*psn*tensor(I,I,H,I,H).dag()
    elif mea_basis == 2:
        psn = tensor(I,I,S*H,I,S*H*X).dag()*psn*tensor(I,I,S*H,I,S*H*X)
    
    meas0 = ket2dm(zero)
    meas1 = zero*one.dag()
    
    meas0 = tensor(I, I, meas0, I, meas0)
    meas1 = tensor(I, I, meas1, I, meas1)

    psn_final = meas0*psn*meas0.dag()+meas1*psn*meas1.dag()
    prob = psn_final.tr()
    psn_final = psn_final.unit()
    
    fid_dist = fidelity(bell_st, psn_final)

    return psn_final, np.real(fid_dist), prob

"""if control == 00, then alice and bob spin is the control
   if control == 01, then alice spin and bob nu is control
   if control == 10, then alice nu and bob spin is control
   if control == 11, then alice and bob nu is control"""
def cnot_distillation(psn, err_rate, control):
    
    cnot_mat = (control%2)*err_cenotn(err_rate) + ((control+1)%2)*err_cnnote(err_rate)
    
    if control == 0:
        cnot_mat_a = err_cenotn(err_rate)
        cnot_mat_b = err_cenotn(err_rate)
    elif control == 1:
        cnot_mat_a = err_cenotn(err_rate)
        cnot_mat_b = err_cnnote(err_rate)
    #elif control == 2:
    #    cnot_mat_a = err_cnnote(err_rate)
    #    cnot_mat_b = err_cenotn(err_rate)
    #elif control == 3:
    #    cnot_mat_a = err_cnnote(err_rate)
    #    cnot_mat_b = err_cnnote(err_rate)

    cnot_mat = tensor(I,cnot_mat_a,cnot_mat_b)

    psn = cnot_mat*psn*cnot_mat.dag()
    return psn    

def distillation(prepared_state, perfect_state, cnot_err):
    fid_list = []
    prob_list = []
    fip_list = []
    
    fid_list1 = []
    prob_list1 = []
    
    psn_list = []
    psn_list1 = []
    op_list = []
    op_list1 = []
    perf_psn_list = []
    perf_psn_list1 = []
    #psn = prepared_state
    """permutation on spin qubit"""
    for perm_num in range(6):
        psn = permutation_distillation(prepared_state, perm_num, 0)
        #perf_psn = permutation_distillation(perfect_state, perm_num, 0)
        op_list.append(perm_num)
        psn_list.append(psn)
        #perf_psn_list.append(perf_psn)
        
        """permutation on nu qubits"""
    for i in range(len(psn_list)):
        for perm_num in range(6):
            psn_list1.append(permutation_distillation(psn_list[i], perm_num, 1))
            #perf_psn_list1.append(permutation_distillation(perf_psn_list[i], perm_num, 1))
            op_list1.append([op_list[i], perm_num])
            
    psn_list.clear()
    perf_psn_list.clear()
    op_list.clear()
    
    """four different types of cnot that can happen"""
    for i in range(len(psn_list1)):
        for cnot_control in range(2):
            for meas_basis in range(3):
                psn_post_cnot = cnot_distillation(psn_list1[i], cnot_err, cnot_control)
                psn1, fid_disti, prob_disti = coinc_distillation_final(psn_post_cnot, meas_basis, cnot_control, cnot_err)
                
                
                op_list.append([op_list1[i], meas_basis, cnot_control])
                fid_list.append(fid_disti)
                prob_list.append(prob_disti)
                fip_list.append(fid_disti*prob_disti)
    
    psn_list1.clear()
    perf_psn_list1.clear()
    op_list1.clear()
    
    zero = basis(2,0)
    one = basis(2,1)
    bell_st = 1/np.sqrt(2)*(tensor(zero,zero,zero,zero,zero) + 
                                     tensor(zero,one,zero,one,zero))

    #for i in range(len(psn_list)):
    #    psn_list1.append(fidelity(psn_list[i], bell_st))
    
    #maxarg = np.argmax(psn_list1)
    
    #fid = psn_list1[maxarg]
    #print(op[2])
    
    #maxarg1 = np.argmax(fip_list)
    #maxarg1 = np.argmax(prob_list)
    maxarg1 = np.argmax(fid_list)
    fidy = fid_list[maxarg1]
    op = op_list[maxarg1]
    #print(op_list[maxarg1])
    #print(op)
    #print(fidy)
    return fidy, prob_list[maxarg1], op

def distillation_operation(prepared_state, operation_list, sqe_err, cnot_err):
    sqe_list = operation_list[0]
    meas = operation_list[1]
    cnot_control = operation_list[2]

    num_qubits = int(np.log2(prepared_state.shape[0]))
    #print(num_qubits)


    psn = permutation_distillation(prepared_state, sqe_list[0], 0)
    if sqe_list[0] != 0:
        psn = depol_channel(psn, sqe_err, 2, num_qubits)
        psn = depol_channel(psn, sqe_err, 4, num_qubits)

    psn = permutation_distillation(psn, sqe_list[1], 1)
    if sqe_list[0] != 0:
        psn = depol_channel(psn, sqe_err, 1, num_qubits)
        psn = depol_channel(psn, sqe_err, 3, num_qubits)

    #psn = depol_channel(psn, sqe_err, 1, num_qubits)

    psn_post_cnot = cnot_distillation(psn, 0, cnot_control)
    psn_post_cnot = depol_channel(psn_post_cnot, cnot_err, 1, num_qubits)
    psn_post_cnot = depol_channel(psn_post_cnot, cnot_err, 2, num_qubits)
    psn_post_cnot = depol_channel(psn_post_cnot, cnot_err, 3, num_qubits)
    psn_post_cnot = depol_channel(psn_post_cnot, cnot_err, 4, num_qubits)
    if meas != 0:
        psn_post_cnot = depol_channel(psn_post_cnot, sqe_err, 2, num_qubits)
        psn_post_cnot = depol_channel(psn_post_cnot, sqe_err, 4, num_qubits)

    psn1, fid_disti, prob_disti = coinc_distillation_final(psn_post_cnot, meas, cnot_control, 0)

    return fid_disti, prob_disti

def dejmps(prepared_state, cnot_err):
    #print(H*S)
    #print((H*S*H)*S*H*S)
    #print(H*Z)
    
    rotation = tensor(I, S.conj()*H, S.conj()*H, S*H, S*H)
    psn = rotation*prepared_state*rotation.dag()
    
    cnot_mat_a = err_cenotn(cnot_err)
    cnot_mat_b = err_cenotn(cnot_err)
    
    psn = tensor(I, cnot_mat_a, cnot_mat_b)*psn*tensor(I, cnot_mat_a, cnot_mat_b).dag()
    
    zero = basis(2,0)
    one = basis(2,1)
    bell_st = 1/np.sqrt(2)*(tensor(zero,zero,zero,zero,zero) + 
                                     tensor(zero,one,zero,one,zero))
    
    meas0 = ket2dm(zero)
    meas1 = zero*one.dag()
    meas0 = tensor(I, I, meas0, I, meas0)
    meas1 = tensor(I, I, meas1, I, meas1)
    
    psn_final = meas0*psn*meas0.dag()+meas1*psn*meas1.dag()
    #print(fidelity(bell_st, (meas0*psn*meas0.dag()).unit()) + fidelity(bell_st, (meas1*psn*meas1.dag()).unit()))
    prob = psn_final.norm()
    psn_final = psn_final.unit()
    
    fid_dist = fidelity(bell_st, psn_final)
    #print(fid_dist)
    
    
    return fid_dist, prob
    
    
def dejmps1(prepared_state, cnot_err):
    #print(H*S)
    #print((H*S*H)*S*H*S)
    #print(H*Z)
    
    rotation = tensor(I, S.conj()*H, S.conj()*H, S*H, S*H)
    psn = rotation*prepared_state*rotation.dag()
    
    cnot_mat_a = err_cenotn(cnot_err)
    cnot_mat_b = err_cnnote(cnot_err)
    
    psn = tensor(I, cnot_mat_a, cnot_mat_b)*psn*tensor(I, cnot_mat_a, cnot_mat_b).dag()
    
    zero = basis(2,0)
    one = basis(2,1)
    bell_st = 1/np.sqrt(2)*(tensor(zero,zero,zero,zero,zero) + 
                                     tensor(zero,one,zero,one,zero))
    
    meas00 = ket2dm(zero)
    meas01 = (plus)
    meas10 = zero*one.dag()
    meas11 = X*H*(minus)
    meas0 = tensor(I, I, meas00, I, meas01)
    meas1 = tensor(I, I, meas10, I, meas11)
    
    psn_final = meas0*psn*meas0.dag()+meas1*psn*meas1.dag()
    prob = psn_final.tr()
    psn_final = psn_final.unit()
    
    fid_dist = fidelity(bell_st, psn_final)
    print(fid_dist)
    print(fidelity(bell_st, meas0*psn*meas0.dag()) + fidelity(bell_st, meas1*psn*meas1.dag()))
    return fid_dist, prob

def bbpsw(prepared_state, cnot_err):
    #print(H*S)
    #print((H*S*H)*S*H*S)
    #print(H*Z)
    
    rotation = tensor(I, S.conj()*H, S.conj()*H, S*H, S*H)
    psn = prepared_state
    
    cnot_mat_a = err_cenotn(cnot_err)
    cnot_mat_b = err_cenotn(cnot_err)
    
    psn = tensor(I, cnot_mat_a, cnot_mat_b)*psn*tensor(I, cnot_mat_a, cnot_mat_b).dag()
    
    zero = basis(2,0)
    one = basis(2,1)
    bell_st = 1/np.sqrt(2)*(tensor(zero,zero,zero,zero,zero) + 
                                     tensor(zero,one,zero,one,zero))
    
    meas0 = ket2dm(zero)
    meas1 = zero*one.dag()
    meas0 = tensor(I, I, meas0, I, meas0)
    meas1 = tensor(I, I, meas1, I, meas1)
    
    psn_final = meas0*psn*meas0.dag()+meas1*psn*meas1.dag()
    prob = psn_final.tr()
    psn_final = psn_final.unit()
    
    fid_dist = fidelity(bell_st, psn_final)
    
    return fid_dist, prob




    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
