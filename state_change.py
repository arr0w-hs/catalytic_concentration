#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 25 14:39:39 2023

@author: hsharma4
"""

import numpy as np
import scipy as sc
import math as math
import matplotlib.pyplot as plt
from qutip import *
from qutip.measurement import measure, measurement_statistics, measure_observable

import sys
import os
sys.path.append(os.path.abspath("//Users/hsharma4/Desktop/Multipartite state concentration/GHZ state project/robustness"))
from locc_base import *
from slocc_base import *

def vec2dm(vector):
    vector1 = np.zeros(len(vector))
    for i in range(len(vector)):
       vector1[i] = np.sqrt(vector[i])
    dm = np.outer(vector1, np.atleast_2d(np.conjugate(vector1)).T)
    #print(dm)
    return dm

def fidelity_func_state(ideal_state, real_state):
    fid = 0
    ideal_state1 = np.zeros(len(ideal_state))
    real_state1 = np.zeros(len(ideal_state))
    for i in range(len(ideal_state)):
       ideal_state1[i] = np.sqrt(ideal_state[i])
       real_state1[i] = np.sqrt(real_state[i])
    
    fid = np.inner(ideal_state1, real_state1)
    #print(fid)
    fid = fid*np.transpose(np.conj(fid))
    #print(fid)   
    return fid

def fidelity_func_dm(ideal_state, density_mat):
    fid = 0
    ideal_state1 = np.zeros(len(ideal_state))
    for i in range(len(ideal_state)):
       ideal_state1[i] = np.sqrt(ideal_state[i])
    #print(ideal_state)
    fid = np.matmul(ideal_state1, density_mat)
    #print(fid)
    fid = np.matmul(fid, np.atleast_2d(np.conjugate(ideal_state1)).T)
    return fid

def trace_dist(ideal_state_dm, density_mat):
    dif = ideal_state_dm - density_mat
    eign_vals = np.linalg.eigvals(dif)
    eign_vals = abs(eign_vals)
    
    return 0.5*np.sum(eign_vals)
    


def state_transformation(final_state, initial_state, erroneous_initial_state):
    final_state = concat_zeros(final_state, initial_state)
    check = 0
    
    """use function for gamma to find the ideal gamma that is needed"""
    gamma_ideal = func_for_gamma(final_state, initial_state)
    gamma_ideal = gamma_ideal/np.sum(gamma_ideal)
    
    diff = gamma_ideal-initial_state
    for i in range(len(diff)):
       if diff[i] <= 1e-6 and diff[i] >= -1*1e-6:
            diff[i] = 20
            gamma_ideal[i] = initial_state[i]
    gamma_ideal = gamma_ideal/np.sum(gamma_ideal)
    majorisation_check = check_majorisation(gamma_ideal, initial_state)
    
    """use function for locc povms to find the ideal povms to get to gamma"""
    povm_out_list, prob_out_list, perm_out_list = locc_povm_func(gamma_ideal, initial_state)
    
    """apply these povms to imperfect state"""
    gamma_obtained_list, prob_obtained_list = locc_povm_on_state(erroneous_initial_state,  povm_out_list, perm_out_list)
    gamma_density_mat = 0
    
    if len(gamma_obtained_list) == 0:
        sum1 = 0
        sum2 = 0
        raise Exception ("gamma failed to obtained")
        
    for i in range(len(gamma_obtained_list)):
        gamma_density_mat += prob_obtained_list[i]*vec2dm(gamma_obtained_list[i])

    """use function for slocc povms to find the ideal povms to get to final state"""
    meas_mat, waste_mat = slocc_povm_func(final_state, initial_state)
    
    """for each gamma density matrix, apply the slocc povm to get output states"""
    probab = 0
    fidelity_total = 0
    output_density_mat, probab = slocc_povm_on_state2(gamma_density_mat, meas_mat)
    
    fidelity_total = fidelity_func_dm(final_state, output_density_mat)
    
    return output_density_mat, fidelity_total, probab

def state_transformation_td(final_state, initial_state, erroneous_initial_state):
    final_state = concat_zeros(final_state, initial_state)
    check = 0
    
    """use function for gamma to find the ideal gamma that is needed"""
    gamma_ideal = func_for_gamma(final_state, initial_state)
    gamma_ideal = gamma_ideal/np.sum(gamma_ideal)
    
    diff = gamma_ideal-initial_state
    for i in range(len(diff)):
       if diff[i] <= 1e-6 and diff[i] >= -1*1e-6:
            diff[i] = 20
            gamma_ideal[i] = initial_state[i]
    gamma_ideal = gamma_ideal/np.sum(gamma_ideal)
    majorisation_check = check_majorisation(gamma_ideal, initial_state)
    
    """use function for locc povms to find the ideal povms to get to gamma"""
    povm_out_list, prob_out_list, perm_out_list = locc_povm_func(gamma_ideal, initial_state)
    
    """apply these povms to imperfect state"""
    gamma_obtained_list, prob_obtained_list = locc_povm_on_state(erroneous_initial_state,  povm_out_list, perm_out_list)
    gamma_density_mat = 0
    
    if len(gamma_obtained_list) == 0:
        sum1 = 0
        sum2 = 0
        raise Exception ("gamma failed to obtained")
        
    for i in range(len(gamma_obtained_list)):
        gamma_density_mat += prob_obtained_list[i]*vec2dm(gamma_obtained_list[i])

    """use function for slocc povms to find the ideal povms to get to final state"""
    meas_mat, waste_mat = slocc_povm_func(final_state, initial_state)
    
    """for each gamma density matrix, apply the slocc povm to get output states"""
    probab = 0
    fidelity_total = 0
    output_density_mat, probab = slocc_povm_on_state2(gamma_density_mat, meas_mat)
    
    final_state_dm = vec2dm(final_state)
    trace_distance = trace_dist(final_state_dm, output_density_mat)
    
    return output_density_mat, trace_distance, probab


def state_transformation2(final_state, initial_state, erroneous_initial_state):
    final_state = concat_zeros(final_state, initial_state)
    initial_state = initial_state/np.sum(initial_state)
    check = 0
    
    """use function for gamma to find the ideal gamma that is needed"""
    gamma_ideal = func_for_gamma(final_state, initial_state)
    gamma_ideal = gamma_ideal/np.sum(gamma_ideal)
    
    diff = gamma_ideal-initial_state
    #for i in range(len(diff)):
       #if diff[i] <= 1e-6 and diff[i] >= -1*1e-6:
            #diff[i] = 20
            #gamma_ideal[i] = initial_state[i]
    #gamma_ideal = gamma_ideal/np.sum(gamma_ideal)
    majorisation_check = check_majorisation(gamma_ideal, initial_state)
    
    count = 0
    #while not majorisation_check and count <=10:
        #count += 1
        #gamma_ideal = func_for_gamma(final_state, initial_state)
        #gamma_ideal = gamma_ideal/np.sum(gamma_ideal)
        #initial_state = initial_state/np.sum(initial_state)
        
        #diff = gamma_ideal-initial_state
        #print(diff)
        #for i in range(len(diff)):
           #if diff[i] <= 1e-6 and diff[i] >= -1*1e-6:
                #diff[i] = 20
                #gamma_ideal[i] = initial_state[i]
        #gamma_ideal = gamma_ideal/np.sum(gamma_ideal)
        
        #majorisation_check = check_majorisation(gamma_ideal, initial_state)
        
        #if count == 10:
            #majorisation_check = check_majorisation(gamma_ideal, initial_state)
            #print("failed to majorise")
            #print(diff)
            #print(np.sum(gamma_ideal))
            #print()
            #raise Exception ("gamma ideal does not majorise initial state")
    
    
    
    #if np.sum(diff) == 20*len(diff):
        #gamma_ideal = initial_state        
    #print(np.sum(gamma_ideal, axis = 0))      
    #print('gamma ideal', gamma_ideal)
    """use function for locc povms to find the ideal povms to get to gamma"""
    povm_out_list, prob_out_list, perm_out_list = locc_povm_func(gamma_ideal, initial_state)
    #print('povm_out_list', len(povm_out_list))
    """apply these povms to imperfect state"""
    gamma_obtained_list, prob_obtained_list = locc_povm_on_state(erroneous_initial_state,  povm_out_list, perm_out_list)
    #print('gamma_obtained_list', len(gamma_obtained_list))
    """use function for slocc povms to find the ideal povms to get to final state"""
    meas_mat, waste_mat = slocc_povm_func(final_state, initial_state)
    #print(meas_mat)
    #print(waste_mat)
    
    """for each gamma obtained from locc povms, apply the slocc povm to get output states"""
    fidelity_total = 0
    probab = 0
    gamma_density_mat = 0
    
    if len(gamma_obtained_list) == 0:
        sum1 = 0
        sum2 = 0
        #for i in range(len(gamma_ideal)):
            #sum1 += gamma_ideal[i]
            #sum2 += initial_state[i]
            #print(sum1, sum2, sum1-sum2)
        
        """
        print("gamma_ideal", gamma_ideal)
        print("inital state", initial_state)
        gamma_ideal = func_for_gamma(final_state, initial_state)
        print(k_t_transform(gamma_ideal, initial_state))
        print("gamma_ideal", gamma_ideal)
        povm_out_list, prob_out_list, perm_out_list = locc_povm_func(gamma_ideal, initial_state)
        print(create_ds_matrix(gamma_ideal, initial_state))
        gamma_obtained_list, prob_obtained_list = locc_povm_on_state(erroneous_initial_state,  povm_out_list, perm_out_list)
        print(len(gamma_obtained_list))
        #print("gamma obtained is null vector")
        
        #print("initial_state", initial_state)
        print(gamma_ideal - initial_state)
        print('povm_out_list', len(povm_out_list))
        print('gamma_obtained_list', len(gamma_obtained_list))
        print(fidelity_total, probab)
        #return np.zeros((len(initial_state), len(initial_state))), fidelity_total, probab
        """
        raise Exception ("gamma failed to obtained")
        #continue
        
    for i in range(len(gamma_obtained_list)):
        gamma_density_mat += prob_obtained_list[i]*vec2dm(gamma_obtained_list[i])
    #print(vec2dm(gamma_obtained_list[i]))    
    #print("gamma_density_mat", gamma_density_mat)
    
    output_density_mat, probab = slocc_povm_on_state2(gamma_density_mat, meas_mat)
    
    
    fidelity_total = fidelity_func_dm(final_state, output_density_mat)
    
    return output_density_mat, fidelity_total, probab
