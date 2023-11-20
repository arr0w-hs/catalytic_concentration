#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 18:25:33 2023

@author: hsharma4
"""

import numpy as np
import scipy as sc
from qutip import *
from qutip.measurement import measure, measurement_statistics, measure_observable

#func for tensor product of a state with itself
def self_tensor_prod(vec, num_of_copies):
    dim = np.shape(vec)[0]
    prod_vec = vec
    for i in range(num_of_copies-1):
        prod_vec = np.reshape(np.tensordot(prod_vec, vec, 0), dim**(i+2))
    prod_vec = np.sort(prod_vec)[::-1]

    return prod_vec

def self_tensor_prod_ver2(vec, num_of_copies):
    dim = np.shape(vec)[0]
    prod_vec = vec
    for i in range(num_of_copies-1):
        prod_vec = np.reshape(np.tensordot(prod_vec, vec, 0), dim**(i+2))
    #prod_vec = np.sort(prod_vec)[::-1]
    
    return prod_vec

#function for calculating entanglement monotone from m onwards
def ent_mono2(vec, m_begin, m_end):
    em = 0
    if m_begin == m_end:
        em = vec[m_begin]
    else:    
        for i in range(m_begin, m_end):
            em += vec[i]
    return em

def concat_zeros(out_state, in_state):
    if np.shape(in_state)[0] >= np.shape(out_state)[0]:
        extra_zeros = np.zeros(np.shape(in_state)[0] - np.shape(out_state)[0])
        out_state = np.concatenate((out_state, extra_zeros), axis=None)
    else:
        raise Exception("Incoherent dimensions of states")
    
    return out_state

def prob_of_transformation(op_state, ip_state):
    if np.shape(ip_state)[0] < np.shape(op_state)[0]:
        raise Exception("Incoherent dimensions of states")

    op_state = concat_zeros(op_state, ip_state)
    p = np.zeros(np.shape(ip_state)[0])
    for i in range(np.shape(ip_state)[0]):
        if ent_mono2(op_state, i, np.shape(ip_state)[0]) != 0:
            p[i] = ent_mono2(ip_state, i, np.shape(ip_state)[0])/ent_mono2(op_state, i, np.shape(ip_state)[0])
        else:
            p[i] = 1000
            #raise Exception("zero probability in denominator of prob of transformation")
    return np.min(p)

def majorisation_check(final_state, initial_state):
    if np.shape(initial_state)[0] < np.shape(final_state)[0]:
        raise Exception("Incoherent dimensions of states")

    final_state = concat_zeros(final_state, initial_state)
    probab = prob_of_transformation(final_state, initial_state)
    if probab < 1:
        return 0
    else:
        return 1

def func_for_lr(opstate, ipstate):
    n = np.shape(ipstate)[0]
    l = []
    r = []
    p = np.zeros(n)
    k = n+1
    l.append(k)
    
    while k>1:
        p = np.zeros(k-1)
        for i in range(k-1):
            if ent_mono2(opstate, i, k-1) != 0:
                p[i] = ent_mono2(ipstate, i, k-1)/ent_mono2(opstate, i, k-1)
            else:
                #print(opstate, i, k-1)
                #raise Exception("zero probability in denominator")
                p[i] = 1000
        m = np.min(p)
        r.append(m)
        l_temp = []
        for i in range(k-1):
            if p[i] == m:
                l_temp.append(i+1)
        l.append(np.min(l_temp))
        k = np.min(l_temp)
    #print(l, r)
    return l, r

def func_for_gamma(opstate, ipstate):
    if np.shape(ipstate)[0] < np.shape(opstate)[0]:
        raise Exception("Incoherent dimensions of states")

    opstate = concat_zeros(opstate, ipstate)
    #print(opstate)
    l_array, r_array = func_for_lr(opstate, ipstate)
    len_r = len(r_array)
    gamma_f = []
    for i in range(len_r):
        i = len_r-1-i
        for j in range(l_array[i+1], l_array[i]):
            gamma_f.append(r_array[i]*opstate[j-1])#np.sqrt(r[i]*ops[j-1]))
    #print(gamma_f)
    return gamma_f

def slocc_povm_func_ver0(opstate, ipstate):
    if np.shape(ipstate)[0] < np.shape(opstate)[0]:
        raise Exception("Incoherent dimensions of states")

    opstate = concat_zeros(opstate, ipstate)
    
    m1 = []
    l_array, r_array = func_for_lr(opstate, ipstate)
    gamma_2 = func_for_gamma(opstate, ipstate)
    len_r = len(r_array)
    for i in range(len_r):
        for j in range(l_array[i]-l_array[i+1]):
            m1.append(r_array[0]/r_array[i])#(np.sqrt(r[0]/r[i])))
    m1.reverse()
    final_opstate = np.multiply(m1, gamma_2)/r_array[0]#np.sqrt(r[0])
    
    return m1, final_opstate, gamma_2, r_array[0]


def slocc_povm_func(op_state, in_state):
    if np.shape(in_state)[0] < np.shape(op_state)[0]:
        raise Exception("Incoherent dimensions of states")

    op_state = concat_zeros(op_state, in_state)
    
    meas_matrix = []
    l_array, r_array = func_for_lr(op_state, in_state)
    gamma = func_for_gamma(op_state, in_state)
    len_r = len(r_array)
    for i in range(len_r):
        for j in range(l_array[i]-l_array[i+1]):
            meas_matrix.append(r_array[0]/r_array[i])#(np.sqrt(r[0]/r[i])))
    meas_matrix.reverse()
    meas_matrix = np.diag(meas_matrix)
    
    waste_matrix = np.eye(len(in_state)) - meas_matrix
    return meas_matrix, waste_matrix

"""gamma new is a density matrix"""
def slocc_povm_on_state(gamma_new_dm, povm_mat):
    
    out_dm = np.matmul(povm_mat, gamma_new_dm)
    #out_dm = np.matmul(out_dm, np.conj(povm_mat))
    prob = np.trace(out_dm)
    
    
    out_dm = out_dm/prob
    #out_state = np.matmul(povm_mat, gamma_new)
    #prob = np.sum(out_state)
    #out_state = out_state/prob
    
    return out_dm, prob

def slocc_povm_on_state2(gamma_new_dm, povm_mat):
    
    out_dm = np.matmul(np.sqrt(povm_mat), gamma_new_dm)
    out_dm = np.matmul(out_dm, np.sqrt(povm_mat))
    prob = np.trace(out_dm)
    
    
    out_dm = out_dm/prob
    #out_state = np.matmul(povm_mat, gamma_new)
    #prob = np.sum(out_state)
    #out_state = out_state/prob
    
    return out_dm, prob
    
