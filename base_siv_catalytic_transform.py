#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:03:48 2023

@author: hsharma4
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

def slocc_povm_on_dm_qobj(gamma_new_dm, povm_mat):
    """function for applying  povm to the gamma obtained in qboj form"""


    povm_mat = Qobj(np.sqrt(povm_mat), dims = [[2,2,2], [2,2,2]])
    povm_mat = tensor(I, povm_mat, I,I,I)

    out_dm = povm_mat* gamma_new_dm* povm_mat.dag()
    prob = 0
    prob = np.trace(out_dm)
    if prob > 1.001:

        print("Error: prob greater than 1")

    if prob != 0:
        out_dm = out_dm/prob
    else:
        raise Exception ("probability is zero")

    return out_dm, prob


def locc_povm_on_dm_qobj(input_state_dm, povm_out_list, perm_out_list):
    """gives out gamma in density matrix form with catalyst"""

    out_states = []
    out_probs = []
    out_state_dm = 0

    for i, element in enumerate(povm_out_list):

        element = Qobj(np.sqrt(element), dims  = [[2,2,2], [2,2,2]])
        element = tensor(I, element, I, I, I)
        perm_out_list[i] = Qobj(perm_out_list[i], dims  = [[2,2,2], [2,2,2]])
        perm_out_list[i] = tensor(I, perm_out_list[i], perm_out_list[i])


        out_dm = element * input_state_dm * (element).dag()
        out_dm = perm_out_list[i].trans() * out_dm * perm_out_list[i]
        prob = out_dm.norm()

        out_state_dm += out_dm
        out_states.append(out_dm)
        out_probs.append(prob)

    return out_state_dm, out_probs

def slocc_povm_on_dm_qobj_nc(gamma_new_dm, povm_mat):
    """function for applying  povm to the gamma obtained in qboj form
    without catalyst"""

    povm_mat = Qobj(np.sqrt(povm_mat), dims = [[2,2], [2,2]])
    povm_mat = tensor(I, povm_mat, I,I)

    out_dm = povm_mat* gamma_new_dm* povm_mat.dag()
    prob = np.trace(out_dm)
    if prob != 0:
        out_dm = out_dm/prob
    else:
        Exception ("probability is zero")
    return out_dm, prob


def locc_povm_on_dm_qobj_nc(input_state_dm, povm_out_list, perm_out_list):
    """gives out gamma in density matrix form without catalyst"""
    out_states = []
    out_probs = []
    out_state_dm = 0

    for i, element in enumerate(povm_out_list):
        element = Qobj(np.sqrt(element), dims  = [[2,2], [2,2]])

        element = tensor(I, element, I, I)
        perm_out_list[i] = Qobj(perm_out_list[i], dims  = [[2,2], [2,2]])
        perm_out_list[i] = tensor(I, perm_out_list[i], perm_out_list[i])

        out_dm = element * input_state_dm * (element).dag()
        out_dm = perm_out_list[i].trans() * out_dm * perm_out_list[i]

        prob = out_dm.norm()

        out_state_dm += out_dm
        out_states.append(out_dm)
        out_probs.append(prob)

    return out_state_dm, out_probs


def schmidt_decomp_of_dm(psn_dm):
    """find the closest pure state, and the schmidt decomposition of it"""
    psn_st, prob = closest_pure_state(psn_dm)
    #print(prob, "prob")
    psn_st, basis_mat, sc = basis2schmidt(psn_st)
    psn_dm = basis_mat * psn_dm * basis_mat.dag()
    schmidt_coeff = np.real((sc.full())**2)
    schmidt_coeff = np.reshape(schmidt_coeff, [len(schmidt_coeff)])

    return schmidt_coeff, psn_dm, psn_st

def prepare_carbon_spins(out_state, input_schmidt_coeff, psn_dm, pure_state):
    """find the optimal catalyst state and convert it into qobj form"""

    pncr, gain, pcr, cat_final, flg = catalytic_concentration(out_state, input_schmidt_coeff, 1, 2)
    carbon_st = (np.sqrt(cat_final[0])*tensor(basis(2,0), basis(2,0))+
                 np.sqrt(cat_final[1])*tensor(basis(2,1), basis(2,1)))
    
    carbon_dm = ket2dm(carbon_st)

    psnc_dm = tensor(carbon_dm, psn_dm)
    pure_state = tensor(carbon_st, pure_state)

    """putting the tensor product in the form |ph>|sp_a>|nu_a>|c>_a|sp_b>|nu_b>|c_b>"""
    pure_state_cat = pure_state.permute([2,3,4,0,5,6,1])
    psnc_dm = psnc_dm.permute([2,3,4,0,5,6,1])
    #print(cat_final, "cat_final")

    return carbon_st, psnc_dm, pure_state_cat, cat_final, gain


def cat_st2cat_array(catalyst_st):
    """converts catalyst state to array of schmidt coefficients"""

    cat_st_matrix_form = np.reshape(catalyst_st, (2, 2))
    U, S, Vh = np.linalg.svd(cat_st_matrix_form, full_matrices=True)
    s_coeff_cat, u, v = Qobj(S), Qobj(U).dag(), Qobj(Vh).dag()

    return np.real((s_coeff_cat.full())**2)

def pre_conversion_process(prepared_dm):
    """function for creating the required arrays and qobjs
    for conversion of states"""

    s_coeff, prepared_dm_schmidt_basis, pure_st = schmidt_decomp_of_dm(prepared_dm)
    s_coeff = np.reshape(s_coeff, [4])
    #print(s_coeff)
    """finding and making the catalyst for the pure statestate"""
    output_states = [0.5, 0.5]
    carbon_st, psnc_dm, psnc_st, cat_array, cat_gain = prepare_carbon_spins(
        output_states, s_coeff, prepared_dm_schmidt_basis, pure_st)
    #print(cat_array)
    psnc_st, basis_mat_cat, ss_cat = basis2schmidt(psnc_st)
    psnc_dm = basis_mat_cat * psnc_dm * basis_mat_cat.dag()
    
    input_state_array = np.reshape(np.real((ss_cat.full())**2), [8])
    output_state_array = np.sort(np.reshape(np.tensordot(output_states, cat_array, 0), 4))[::-1]
    output_state_array = concat_zeros(output_state_array, input_state_array)

    output_state_qobj = 1/np.sqrt(2)*(tensor(zero,zero,zero,zero,zero) +
                                     tensor(zero,zero,one,zero,one))
    output_state_qobj = tensor(carbon_st, output_state_qobj)
    output_state_qobj = ket2dm(output_state_qobj.permute([2,3,0,4,5,1,6]))
    
    return output_state_qobj, psnc_dm, carbon_st, input_state_array, output_state_array

def catalytic_conversion(prepared_dm):
    """function for taking in two prepared bell states and converts them
    by (1) finding the optimal catalyst, (2) finding the povms, and
    (3) applys the povms"""

    output_state_qobj, psnc_dm, carbon_st, input_state_array, output_state_array = pre_conversion_process(prepared_dm)
    
    """use function for gamma to find the ideal gamma that is needed"""
    gamma_ideal = func_for_gamma(output_state_array, input_state_array)
    
    if np.sum(gamma_ideal) != 0:
        gamma_ideal = gamma_ideal/np.sum(gamma_ideal)
    else:
        raise Exception ("gamma ideal sums to zero")
    
    majorisation_check(gamma_ideal, input_state_array)
    
    if np.any(input_state_array == 0):
        print(input_state_array, "input")
        print(gamma_ideal, "gamma")
        print(carbon_st, "carbon")
        print("input state array has a 0")
    """use function for locc povms to find the ideal povms to get to gamma"""
    povm_out_list, prob_out_list_junk, perm_out_list = locc_povm_func(gamma_ideal,
                                                                 input_state_array)
    
    """apply those povm on the density matrix"""
    gamma_density_mat, prob_obtained_list = locc_povm_on_dm_qobj(psnc_dm,
                                                                 povm_out_list, perm_out_list)

    if len(prob_obtained_list) == 0:
        raise Exception ("gamma failed to obtained")

    """use function for slocc povms to find the ideal povms to get to final state"""
    meas_mat, prob_out_list_junk = slocc_povm_func(output_state_array, input_state_array)

    """for each gamma density matrix, apply the slocc povm to get output states"""
    probab = 0
    output_density_mat, probab = slocc_povm_on_dm_qobj(gamma_density_mat, meas_mat)
    
    fid = fidelity(output_state_qobj, output_density_mat)

    return fid, np.real(probab), output_density_mat, carbon_st


def non_catalytic_conversion(prepared_dm):
    """function for taking in two prepared bell states and converts them
    by (1) finding the povms, and
    (2) applys the povms"""

    output_state = [0.5, 0.5]

    s_coeff, prepared_dm_schmidt_basis, pure_st = schmidt_decomp_of_dm(prepared_dm)
    input_state_array = np.reshape(np.real(s_coeff), [4])

    if np.sum(input_state_array) != 0:
        input_state_array = input_state_array/np.sum(input_state_array)
    else:
        raise Exception ("values in input state array sum to zero")

    output_state_array = concat_zeros(output_state, input_state_array)

    output_state_qobj = 1/np.sqrt(2)*(tensor(zero,zero,zero,zero,zero) +
                                     tensor(zero,zero,one,zero,one))


    """use function for gamma to find the ideal gamma that is needed"""
    gamma_ideal = func_for_gamma(output_state_array, input_state_array)
    if np.sum(gamma_ideal) != 0:
        gamma_ideal = gamma_ideal/np.sum(gamma_ideal)
    else:
        raise Exception ("gamma ideal sums to zero")

    """use function for locc povms to find the ideal povms to get to gamma"""
    povm_out_list, prob_out_list_junk, perm_out_list = locc_povm_func(gamma_ideal, input_state_array)
    
    """use function for slocc povms to find the ideal povms to get to final state"""
    meas_mat, prob_out_list_junk = slocc_povm_func(output_state_array, input_state_array)

    """apply those povm on the density matrix"""
    gamma_density_mat, prob_obtained_list = locc_povm_on_dm_qobj_nc(
        prepared_dm_schmidt_basis, povm_out_list, perm_out_list)

    if len(prob_obtained_list) == 0:
        majorisation_check = check_majorisation(gamma_ideal, input_state_array)
        raise Exception ("gamma failed to obtained")

    """for gamma density matrix, apply the slocc povm to get output states"""
    probab = 0
    output_density_mat, probab = slocc_povm_on_dm_qobj_nc(gamma_density_mat, meas_mat)

    fid = fidelity(output_state_qobj, output_density_mat)
    return fid, np.real(probab), output_density_mat


def catalytic_conversion_reuse(prepared_dm, carbon_dm_input):
    """function for taking in two prepared bell states and converts them
    by (1) finding the optimal catalyst, (2) finding the povms, and
    (3) applys the povms, (4) reusing the catalyst"""

    flagg = 0
    s_coeff, prepared_dm_schmidt_basis, pure_st = schmidt_decomp_of_dm(prepared_dm)
    s_coeff = np.reshape(s_coeff, [4])

    """finding and making the catalyst for the pure statestate"""
    output_states = [0.5, 0.5]
    #carbon_st, psnc_dm, psnc_st, cat_array, cat_gain = prepare_carbon_spins(
    #    output_states, s_coeff, prepared_dm_schmidt_basis, pure_st)

    carbon_st, prob = closest_pure_state(carbon_dm_input)

    psnc_dm = tensor(carbon_dm_input, prepared_dm_schmidt_basis)
    psnc_st = tensor(carbon_st, pure_st)

    """putting the tensor product in the form |ph>|sp_a>|nu_a>|c>_a|sp_b>|nu_b>|c_b>"""
    psnc_st = psnc_st.permute([2,3,4,0,5,6,1])
    psnc_dm = psnc_dm.permute([2,3,4,0,5,6,1])

    cat_array = cat_st2cat_array(carbon_st)

    psnc_st, basis_mat_cat, ss_cat = basis2schmidt(psnc_st)
    psnc_dm = basis_mat_cat * psnc_dm * basis_mat_cat.dag()
    
    input_state_array = np.reshape(np.real((ss_cat.full())**2), [8])
    output_state_array = np.sort(np.reshape(np.tensordot(output_states, cat_array, 0), 4))[::-1]
    output_state_array = concat_zeros(output_state_array, input_state_array)
    
    output_state_qobj = 1/np.sqrt(2)*(tensor(zero,zero,zero,zero,zero) +
                                     tensor(zero,zero,one,zero,one))
    output_state_qobj = tensor(carbon_st, output_state_qobj)
    output_state_qobj = (output_state_qobj.permute([2,3,0,4,5,1,6]))
    
    """use function for gamma to find the ideal gamma that is needed"""
    gamma_ideal = func_for_gamma(output_state_array, input_state_array)
    if np.sum(gamma_ideal) != 0:
        gamma_ideal = gamma_ideal/np.sum(gamma_ideal)
    else:
        raise Exception ("gamma ideal sums to zero")

    majorisation_check = check_majorisation(gamma_ideal, input_state_array)

    if np.any(input_state_array <= 1e-12):
        flagg = 1
        output_density_mat = tensor(carbon_dm_input, prepared_dm_schmidt_basis)
        output_density_mat = output_density_mat.permute([2,3,0,4,5,1,6])
        fid = 0
        probab = 0
        print(carbon_st, "carbon")
        print("state too small")

    else:
        """use function for locc povms to find the ideal povms to get to gamma"""
        povm_out_list, prob_out_list, perm_out_list = locc_povm_func(gamma_ideal, input_state_array)

        """apply those povm on the density matrix"""
        gamma_density_mat, prob_obtained_list = locc_povm_on_dm_qobj(psnc_dm,
                                                povm_out_list, perm_out_list)

        if len(prob_obtained_list) == 0:
            raise Exception ("gamma failed to obtained")

        """use function for slocc povms to find the ideal povms to get to final state"""
        meas_mat, waste_mat = slocc_povm_func(output_state_array, input_state_array)

        """for each gamma density matrix, apply the slocc povm to get output states"""
        probab = 0
        output_density_mat, probab = slocc_povm_on_dm_qobj(gamma_density_mat, meas_mat)
        fid = fidelity(output_state_qobj, output_density_mat)
        
    return fid, np.real(probab), output_density_mat, flagg

def pre_conversion_process_reuse(prepared_dm, carbon_dm_input):
    """function for creating the required arrays and qobjs
    for conversion of states"""

    s_coeff, prepared_dm_schmidt_basis, pure_st = schmidt_decomp_of_dm(prepared_dm)
    s_coeff = np.reshape(s_coeff, [4])

    """finding and making the catalyst for the pure statestate"""
    output_states = [0.5, 0.5]
    carbon_st, psnc_dm, psnc_st, cat_array, cat_gain = prepare_carbon_spins(
        output_states, s_coeff, prepared_dm_schmidt_basis, pure_st)

    psnc_st, basis_mat_cat, ss_cat = basis2schmidt(psnc_st)
    psnc_dm = basis_mat_cat * psnc_dm * basis_mat_cat.dag()
    
    input_state_array = np.reshape(np.real((ss_cat.full())**2), [8])
    output_state_array = np.sort(np.reshape(np.tensordot(output_states, cat_array, 0), 4))[::-1]
    output_state_array = concat_zeros(output_state_array, input_state_array)


    output_state_qobj = 1/np.sqrt(2)*(tensor(zero,zero,zero,zero,zero) +
                                     tensor(zero,zero,one,zero,one))
    output_state_qobj = tensor(carbon_st, output_state_qobj)
    output_state_qobj = output_state_qobj.permute([2,3,4,0,5,6,1])

    return output_state_qobj, psnc_dm, input_state_array, output_state_array


"""for converting without any knowledge of catalyst after using it once"""
def catalytic_conversion_reuse2(prepared_dm, carbon_dm_input):
    """function for taking in two prepared bell states and converts them
    by (1) finding the optimal catalyst, (2) finding the povms, and
    (3) applys the povms"""

    junk0, junk, carbon_st, input_state_array, output_state_array = pre_conversion_process(prepared_dm)
    
    psnc_dm = tensor(carbon_dm_input, prepared_dm)
    psnc_dm = psnc_dm.permute([2,3,4,0,5,6,1])
    
    output_state_qobj = ket2dm(1/np.sqrt(2)*(tensor(zero,zero,zero,zero,zero) +
                                     tensor(zero,zero,one,zero,one)))
    output_state_qobj = tensor(carbon_dm_input, output_state_qobj)
    output_state_qobj = output_state_qobj.permute([2,3,4,0,5,6,1])
    
    """use function for gamma to find the ideal gamma that is needed"""
    gamma_ideal = func_for_gamma(output_state_array, input_state_array)
    if np.sum(gamma_ideal) != 0:
        gamma_ideal = gamma_ideal/np.sum(gamma_ideal)
    else:
        raise Exception ("gamma ideal sums to zero")
    
    if np.any(input_state_array == 0):
        print(input_state_array, "input")
        print(gamma_ideal, "gamma")
        print(carbon_st, "carbon")
        print("input state has a 0")
    """use function for locc povms to find the ideal povms to get to gamma"""
    povm_out_list, prob_out_list_junk, perm_out_list = locc_povm_func(gamma_ideal,
                                                                 input_state_array)

    """apply those povm on the density matrix"""
    gamma_density_mat, prob_obtained_list = locc_povm_on_dm_qobj(psnc_dm,
                                                                 povm_out_list, perm_out_list)

    if len(prob_obtained_list) == 0:
        raise Exception ("gamma failed to obtained")

    """use function for slocc povms to find the ideal povms to get to final state"""
    meas_mat, prob_out_list_junk = slocc_povm_func(output_state_array, input_state_array)

    """for each gamma density matrix, apply the slocc povm to get output states"""
    probab = 0
    output_density_mat, probab = slocc_povm_on_dm_qobj(gamma_density_mat, meas_mat)
    fid = fidelity(output_state_qobj, output_density_mat)

    return fid, np.real(probab), output_density_mat, carbon_st
