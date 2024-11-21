#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 18:26:42 2023

@author: hsharma4
"""
import sys
import os

import numpy as np
#from qutip import *
#from qutip.measurement import measure, measurement_statistics, measure_observable
import networkx as nx
from base_slocc import concat_zeros
sys.path.append(os.path.dirname(__file__))



# def check_majorisation(final_state, initial_state):
#     """func for checking majorisation of input and output states"""

#     sumf = 0
#     sumi = 0
#     res = 1
#     for i in range(len(final_state)):
#         sumf += final_state[i]
#         sumi += initial_state[i]
#         if sumf < sumi:
#             res = 0

#     return res

def k_t_transform(final_state, input_state):
    """func for finding value of k and t for constructing D matrix"""

    k_value = 0
    for i in range(len(final_state)-1):
        if final_state[i+1] <= input_state[0] and final_state[i] >= input_state[0]:
            k_value = i+1

    if (final_state[0]-final_state[k_value]) != 0:
        t_value = (input_state[0]-final_state[k_value])/(final_state[0]-final_state[k_value])
    else:
        t_value = 0

    return k_value, t_value


def create_t_matrix(t, j, k, vec):
    """
    j and k belong from 0 to n-1 and are the values of the
    places where T-transform works
    """

    t_matrix = np.eye(len(vec))
    t_matrix[j,j] = t
    t_matrix[k,k] = t
    t_matrix[k,j] = 1-t
    t_matrix[j,k] = 1-t

    return t_matrix

def create_ds_matrix(final_state, input_state):
    """
    output_state is the final state after LOCC transformation
    input state is the inital state for LOCC
    the method of construction of D-matrix goes from
    output state to input state
    """

    count = 0
    assert len(final_state) <= len(input_state)
        #raise Exception("Incoherent dimensions of states")

    ips_temp = input_state
    ops_temp = final_state
    d_matrix = np.eye(len(input_state))
    t_list = []
    while(not np.array_equal(ips_temp, ops_temp) and count <= 1200):

        k, t = k_t_transform(ops_temp, ips_temp)

        d_matrix_temp = create_t_matrix(t, 0, k, ips_temp)
        ops_temp = np.matmul(d_matrix_temp, ops_temp)
        identity_temp = np.eye(count)
        s_temp = len(ops_temp)

        t_matrix = np.block([[identity_temp, np.zeros((count, s_temp))],
                  [np.zeros((s_temp, count)), d_matrix_temp]
                  ])

        d_matrix = np.matmul(d_matrix, t_matrix)

        t_list.append(t_matrix)
        ips_temp = np.delete(ips_temp,0)
        ops_temp = np.delete(ops_temp,0)
        count += 1
        if len(ips_temp) == 1:
            count = 1210

    return np.transpose(d_matrix), t_list

def create_ds_matrix2(final_state, input_state):
    """
    output_state is the final state after LOCC transformation
    input state is the inital state for LOCC
    the method of construction of D-matrix goes from
    output state to input state
    """

    count = 0
    assert len(final_state) <= len(input_state)
    #    raise Exception("Incoherent dimensions of states")

    ips_temp = input_state
    ops_temp = final_state
    d_matrix = np.eye(len(input_state))
    t_list = []
    while(not np.array_equal(ips_temp, ops_temp) and count <= 1200):

        k, t = k_t_transform(ops_temp, ips_temp)

        d_matrix_temp = create_t_matrix(t, 0, k, ips_temp)
        ops_temp = np.matmul(d_matrix_temp, ops_temp)
        t_list.append(d_matrix_temp)
        s_temp = len(ops_temp)
        A = np.eye(count)
        d_matrix = np.matmul(d_matrix,
                             np.block([[A, np.zeros((count, s_temp))],
                                       [np.zeros((s_temp, count)), d_matrix_temp]
                                       ])
                             )

        ips_temp = np.delete(ips_temp,0)
        ops_temp = np.delete(ops_temp,0)
        count += 1
        if len(ips_temp) == 1:
            count = 1210

    return np.transpose(d_matrix), t_list

def create_adj_mat(ds_mat):
    """funtion for creating adjacency matrix from D matrix"""

    dim = np.shape(ds_mat)[0]

    dim_adj = 2*dim
    adj_mat = np.zeros([dim_adj, dim_adj])
    for i in range(dim):
        for j in range(dim):
            if ds_mat[j,i] > 0:
                adj_mat[i,dim+j] = 1
                adj_mat[dim+j, i] = 1
    #print(adj_mat)

    return adj_mat

def create_matching(adj):
    """create matching from the adj matrix"""

    graph = nx.from_numpy_array(adj)
    matching = list(nx.min_weight_matching(graph))

    return matching

def match2perm(ds_mat):
    """converting matching to permutation matrix"""

    dim = np.shape(ds_mat)[0]
    adj = create_adj_mat(ds_mat)
    matching = create_matching(adj)

    perm = np.zeros([dim, dim])
    for i in range(len(matching)):
        a = list(matching[i])
        a = np.sort(a)

        if a[1] > dim-1:
            a[1] = a[1] - dim
        if a[0] > dim-1:
            a[0] = a[0] - dim

        perm[a[1], a[0]] = 1
    sum1 = np.sum(perm, axis = 0)
    sum2 = np.sum(perm, axis = 1)
    for i in range(dim):
        assert sum1[i] == 1 and  sum2[i] == 1


    return perm

def minVal_permMat(ds_mat):
    """probability of permutation matrix"""

    min_value = 0
    count = 0
    while min_value == 0 and count <1000:
        count += 1

        perm_mat = match2perm(ds_mat)
        mat = np.multiply(ds_mat, perm_mat)
        mat = np.sum(mat, axis=0)
        min_value = np.min(mat)

    return min_value, perm_mat

def permutation_mat_list(ds_mat):
    """function for creating list of permutation matrices"""

    count = 0
    sum_values = np.sum(ds_mat, axis=0)
    sum_values = np.sum(sum_values, axis=0)
    minval_list = []
    perm_mat_list = []
    while sum_values > 10e-8 and count < 1000:
        count += 1
        minval, perm_mat = minVal_permMat(ds_mat)

        minval_list.append(minval)
        perm_mat_list.append(perm_mat)

        ds_mat = ds_mat - minval*perm_mat
        sum_values = np.sum(ds_mat, axis=0)
        sum_values = np.sum(sum_values, axis=0)

    return perm_mat_list, minval_list

def locc_povm_func(final_state, ini_state):
    """function for creating list of povm and permutation matrices"""

    final_state = concat_zeros(final_state, ini_state)

    #maj_cehck = check_majorisation(final_state, ini_state)
    ds, _ = create_ds_matrix(final_state, ini_state)
    perm_list, prob_list =  permutation_mat_list(ds)
    povm_list = []
    num_perm_mat = len(prob_list)
    dim_vec = len(final_state)


    for i in range(num_perm_mat):
        povm = np.zeros((dim_vec, dim_vec))
        beta = 0
        beta = np.matmul(perm_list[i], final_state)

        if np.any(ini_state == 0):
            raise Exception("initial state has a zero")

        beta = beta/ini_state
        if np.any(np.isnan(beta)):
            print("error beta has nan")

        beta = prob_list[i]*beta

        np.fill_diagonal(povm, beta)
        povm_list.append(povm)
        #print(np.linalg.matrix_rank(povm))
        #print(np.nonzero(povm))
    #print("ed")

    return povm_list, prob_list, perm_list


def locc_povm_on_state(new_state, povm_out_list, perm_out_list):
    """function for applying povm on errored input state"""

    out_states = []
    out_probs = []

    for i in range(len(povm_out_list)):
        out_state = np.matmul(povm_out_list[i], new_state)

        rho = np.diag(new_state)

        out = np.matmul(povm_out_list[i], rho)
        out = np.trace(out)
        out_probs.append(out)

        out_state = out_state/out
        out_state = np.matmul(np.transpose(perm_out_list[i]), out_state)
        out_states.append(out_state)

    return out_states, out_probs
