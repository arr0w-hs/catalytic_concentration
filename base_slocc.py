#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 18:25:33 2023

@author: hsharma4
"""

import numpy as np
from qutip import *
#from qutip.measurement import measure, measurement_statistics, measure_observable


def self_tensor_prod(vec, num_of_copies):
    """func for tensor product of a state with itself"""

    dim = np.shape(vec)[0]
    prod_vec = vec
    for i in range(num_of_copies-1):
        prod_vec = np.reshape(np.tensordot(prod_vec, vec, 0), dim**(i+2))
    prod_vec = np.sort(prod_vec)[::-1]

    return prod_vec


def self_tensor_prod_ver2(vec, num_of_copies):
    """func for tensor product of a state with itself version 2"""

    dim = np.shape(vec)[0]
    prod_vec = vec
    for i in range(num_of_copies-1):
        prod_vec = np.reshape(np.tensordot(prod_vec, vec, 0), dim**(i+2))

    return prod_vec


def ent_mono2(vec, m_begin, m_end):
    """function for calculating entanglement monotone from m_begin onwards"""

    ent_monotone = 0
    if m_begin == m_end:
        ent_monotone = vec[m_begin]
    else:
        for i in range(m_begin, m_end):
            ent_monotone += vec[i]

    return ent_monotone

def concat_zeros(out_state, in_state):
    """func for concating zeros in output state"""

    if np.shape(in_state)[0] >= np.shape(out_state)[0]:
        extra_zeros = np.zeros(np.shape(in_state)[0] - np.shape(out_state)[0])
        out_state = np.concatenate((out_state, extra_zeros), axis=None)
    else:
        raise Exception("Incoherent dimensions of states")
    
    #out_state = np.sort(out_state)[::-1]
    
    return out_state

def prob_of_transformation(op_state, ip_state):
    """func for finding the probability of transformation"""

    if np.shape(ip_state)[0] < np.shape(op_state)[0]:
        raise Exception("Incoherent dimensions of states")

    probab_array = np.zeros(np.shape(ip_state)[0])
    for i in range(np.shape(ip_state)[0]):
        if ent_mono2(op_state, i, np.shape(ip_state)[0]) != 0:
            probab_array[i] = (ent_mono2(ip_state, i, np.shape(ip_state)[0])/
                    ent_mono2(op_state, i, np.shape(ip_state)[0]))
        else:
            probab_array[i] = 1000

    return np.min(probab_array)

def majorisation_check(final_state, initial_state):
    """func for checking majorisation"""

    if np.shape(initial_state)[0] < np.shape(final_state)[0]:
        raise Exception("Incoherent dimensions of states")

    probab = prob_of_transformation(final_state, initial_state)
    if probab < 0.999:
        print("failed to majorise")

    return


def func_for_lr(opstate, ipstate):
    """func for finding l and r values"""

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
                p[i] = 1000
        m = np.min(p)
        r.append(m)
        l_temp = []
        for i in range(k-1):
            if p[i] == m:
                l_temp.append(i+1)
        l.append(np.min(l_temp))
        k = np.min(l_temp)

    return l, r

def func_for_gamma(opstate, ipstate):
    """func for creating gamma vector"""

    if np.shape(ipstate)[0] < np.shape(opstate)[0]:
        raise Exception("Incoherent dimensions of states")

    l_array, r_array = func_for_lr(opstate, ipstate)
    len_r = len(r_array)
    gamma_f = []
    for i in range(len_r):
        i = len_r-1-i
        for j in range(l_array[i+1], l_array[i]):
            gamma_f.append(r_array[i]*opstate[j-1])

    return gamma_f

def slocc_povm_func_ver0(opstate, ipstate):
    """first version of SLOCC POVM function"""

    if np.shape(ipstate)[0] < np.shape(opstate)[0]:
        raise Exception("Incoherent dimensions of states")

    m1 = []
    l_array, r_array = func_for_lr(opstate, ipstate)
    gamma_2 = func_for_gamma(opstate, ipstate)
    len_r = len(r_array)
    for i in range(len_r):
        for j in range(l_array[i]-l_array[i+1]):
            m1.append(r_array[0]/r_array[i])
    m1.reverse()
    final_opstate = np.multiply(m1, gamma_2)/r_array[0]

    return m1, final_opstate, gamma_2, r_array[0]


def slocc_povm_func(op_state, in_state):
    """SLOCC POVM function"""

    if np.shape(in_state)[0] < np.shape(op_state)[0]:
        raise Exception("Incoherent dimensions of states")

    meas_matrix = []
    l_array, r_array = func_for_lr(op_state, in_state)
    #gamma = func_for_gamma(op_state, in_state)
    len_r = len(r_array)
    for i in range(len_r):
        for j in range(l_array[i]-l_array[i+1]):
            meas_matrix.append(r_array[0]/r_array[i])
    meas_matrix.reverse()
    meas_matrix = np.diag(meas_matrix)

    waste_matrix = np.eye(len(in_state)) - meas_matrix

    return meas_matrix, waste_matrix


def slocc_povm_on_state(gamma_new_dm, povm_mat):
    """gamma new is a density matrix"""

    out_dm = np.matmul(povm_mat, gamma_new_dm)
    prob = np.trace(out_dm)

    out_dm = out_dm/prob

    return out_dm, prob

def slocc_povm_on_state2(gamma_new_dm, povm_mat):
    """gamma new is a density matrix"""

    out_dm = np.matmul(np.sqrt(povm_mat), gamma_new_dm)
    out_dm = np.matmul(out_dm, np.sqrt(povm_mat))
    prob = np.trace(out_dm)

    out_dm = out_dm/prob

    return out_dm, prob
