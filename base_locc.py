#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 11 18:26:42 2023

@author: hsharma4
"""

import numpy as np
import scipy as sc
from qutip import *
from qutip.measurement import measure, measurement_statistics, measure_observable
import networkx as nx

import sys
import os
sys.path.append(os.path.dirname(__file__))
from base_slocc import *

def concat_zeros(out_state, in_state):
    if np.shape(in_state)[0] >= np.shape(out_state)[0]:
        extra_zeros = np.zeros(np.shape(in_state)[0] - np.shape(out_state)[0])
        out_state = np.concatenate((out_state, extra_zeros), axis=None)
    else:
        raise Exception("Incoherent dimensions of states")
    
    return out_state

def check_majorisation(final_state, initial_state):
    sumf = 0
    sumi = 0
    res = 1
    for i in range(len(final_state)):
        sumf += final_state[i]
        sumi += initial_state[i]
        if sumf < sumi:
            #print("fail maj")
            #print(sumf, "sumf")
            #print(sumi, "sumi")
            #print(sumf-sumi, "diff")
            res = 0 
            #print(final_state, "gamma")
            #print(initial_state, "ini state")
            #raise Exception ("gamma does not majorise")
            #print(sumf, sumi, 'i', i)
            #print(np.sum(final_state), np.sum(initial_state))
            #print(i)
            #print(final_state)
            #print(initial_state)
            #print("failed to majorise")
    return res

def k_t_transform(final_state, input_state):
    k_value = 0
    for i in range(len(final_state)-1):
        if final_state[i+1] <= input_state[0] and final_state[i] >= input_state[0]:
            k_value = i+1
        #else:
            #print(final_state[i], final_state[i+1], input_state[0])
            #k_value = len(final_state)-1
    
    #print(k_value, "k_value")
    #print((final_state[0]-final_state[k_value]), "final state diff")
    #print((input_state[0]-final_state[k_value]), "ini state diff")
    
    if (final_state[0]-final_state[k_value]) != 0:
        t_value = (input_state[0]-final_state[k_value])/(final_state[0]-final_state[k_value])
        #for i in range(len(final_state)-1):
        #   print(final_state[i], final_state[i+1], input_state[0])
    else:
        t_value = 0
        
    #print(k_value, t_value)
    return k_value, t_value

#j and k belong from 0 to n-1 and are the values of the 
#places where T-transform works
def create_t_matrix(t, j, k, vec):
    t_matrix = np.eye(len(vec))
    t_matrix[j,j] = t
    t_matrix[k,k] = t
    t_matrix[k,j] = 1-t
    t_matrix[j,k] = 1-t
    
    return t_matrix

#output_state is the final state after LOCC transformation
#input state is the inital state for LOCC
#the method of construction of D-matrix goes from 
#output state to input state
def create_ds_matrix(final_state, input_state):
    count = 0
    if len(final_state) > len(input_state):
        raise Exception("Incoherent dimensions of states")
    
    ips_temp = input_state
    ops_temp = final_state
    d_matrix = np.eye(len(input_state))
    while(not np.array_equal(ips_temp, ops_temp) and count <= 1200):
        
        k, t = k_t_transform(ops_temp, ips_temp)
        #print(k, t)
        d_matrix_temp = create_t_matrix(t, 0, k, ips_temp)
        ops_temp = np.matmul(d_matrix_temp, ops_temp)
        
        s_temp = len(ops_temp)
        A = np.eye(count)
        d_matrix = np.matmul(d_matrix, 
                             np.block([[A, np.zeros((count, s_temp))],[np.zeros((s_temp, count))
                                                                       , d_matrix_temp]]))
        
        ips_temp = np.delete(ips_temp,0)
        ops_temp = np.delete(ops_temp,0)
        count += 1
        if len(ips_temp) == 1:
            count = 1210
    
    return np.transpose(d_matrix)

def create_adj_mat(ds_mat):
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
    
    graph = nx.from_numpy_matrix(adj)
    #print("creating_matching")
    #matching = list(nx.maximal_matching(graph))
    matching = list(nx.min_weight_matching(graph))
    
    #print("created_matching")
    return matching

def match2perm(ds_mat):
    dim = np.shape(ds_mat)[0]
    adj = create_adj_mat(ds_mat)  
    matching = create_matching(adj)
    #print(matching)
    perm = np.zeros([dim, dim])
    for i in range(len(matching)):
        a = list(matching[i])
        a = np.sort(a)
       # print(a)
        if a[1] > dim-1:
            a[1] = a[1] - dim
        if a[0] > dim-1:
            a[0] = a[0] - dim    
        #print(a)
        
        perm[a[1], a[0]] = 1
    sum1 = np.sum(perm, axis = 0)
    sum2 = np.sum(perm, axis = 1)
    #print(perm)
    #print(adj)
    for i in range(dim):
        if sum1[i] != 1 or sum2[i] != 1:
            raise Exception("not perm")
    return perm

def minVal_permMat(ds_mat):
    min_value = 0
    count = 0
    while min_value == 0 and count <1000:
        count += 1
        #perm_mat = make_permutation_matrix(ds_mat)
        perm_mat = match2perm(ds_mat)
        mat = np.multiply(ds_mat, perm_mat)
        mat = np.sum(mat, axis=0)
        min_value = np.min(mat)
        
    return min_value, perm_mat

def permutation_mat_list(ds_mat):
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

def locc_povm_func(final_state, ini_state):#, test_flag):
    #maj_cehck = check_majorisation(final_state, ini_state)
    ds = create_ds_matrix(final_state, ini_state)
    perm_list, prob_list =  permutation_mat_list(ds)
    povm_list = []
    num_perm_mat = len(prob_list)
    dim_vec = len(final_state)
    #if test_flag == 1:
    #    print("ds size", np.shape(ds))
    #    print("ds")
    #    print(ds)
    #    print(len(perm_list))
        
    for i in range(num_perm_mat):
        povm = np.zeros((dim_vec, dim_vec))
        #print(perm_list[i])
        beta = 0
        beta = np.matmul(perm_list[i], final_state)
        #print(ini_state)
        
        if np.any(ini_state == 0):
            #for j in range(len(ini_state)):
            #    if ini_state[j] != 0:
            #        beta[j] = beta[j]/ini_state[j]
            #    else:
            #        beta[j] = 0
            raise Exception("initial state has a zero")
        else: 
            beta = beta/ini_state
        if np.any(np.isnan(beta)):
            print("error beta has nan")
        
        beta = prob_list[i]*beta

        
        #print(beta)
        
        np.fill_diagonal(povm, beta)
        #print(povm)
        povm_list.append(povm)
        #print(povm)
    return povm_list, prob_list, perm_list


def locc_povm_on_state(new_state, povm_out_list, perm_out_list):
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

