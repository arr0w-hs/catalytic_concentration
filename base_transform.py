#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 19:45:19 2024

@author: hsharma4
"""

import sys
import os
import numpy as np
import qutip as qt

from base_locc import locc_povm_func
from base_slocc import concat_zeros, func_for_gamma, majorisation_check, slocc_povm_func
from base_catalyst import catalytic_concentration

sys.path.append(os.path.dirname(__file__))

zero = qt.basis(2,0)
one = qt.basis(2,1)
I = qt.qeye(2)

def closest_pure_state(density_mat):
    """func for finding closest pure state"""
    eigen_array = density_mat.eigenstates()
    #print(eigen_array)
    eigen_states = np.asarray(eigen_array[1])
    eigen_values = np.asarray(eigen_array[0])

    max_arg = np.argmax(eigen_values)
    max_eigenvalue = eigen_values[max_arg]
    max_eigenvector = eigen_states[max_arg]
    #")
    return max_eigenvector, max_eigenvalue


def basis2schmidt(psn_st):
    """converting pure state to schmidt basis"""
    psn_dims = psn_st.dims
    psn_data = psn_st.full()
    """removing the photon state, as it is already in |0>"""
    sn_data = np.array_split(psn_data, 2)[0]
    psn_length = sn_data.shape[0]
    psn_length = int(np.sqrt(psn_length))
    sn_matrix_form = np.reshape(sn_data, (psn_length, psn_length))

    u_mat, s_mat, v_herm_mat = np.linalg.svd(sn_matrix_form, full_matrices=True)
    #print(np.allclose(sn_matrix_form, np.dot(U* S, Vh)))
    s_mat, u_mat, v_mat = qt.Qobj(s_mat), qt.Qobj(u_mat).dag(), qt.Qobj(v_herm_mat).dag()
    basis_matrix = qt.tensor( I, u_mat, v_mat.trans()).full()
    basis_matrix = qt.Qobj(basis_matrix, dims = [psn_dims[0], psn_dims[0]])

    psn = psn_st.full()
    psn = qt.Qobj(psn, dims = psn_dims)#
    psn_schmidt_basis = (basis_matrix*psn).tidyup()
    #psn_schmidt_basis = psn_schmidt_basis.full()
    #psn_schmidt_basis = Qobj(psn_schmidt_basis, dims = psn_dims)

    return psn_schmidt_basis, basis_matrix, s_mat, u_mat, v_mat.trans()

def schmidt_decomp_of_dm(psn_dm):
    """find the closest pure state, and the schmidt decomposition of it"""
    psn_st, _ = closest_pure_state(psn_dm)
    #print(prob, "prob")
    psn_st, basis_mat, sc_mat, u_mat, v_mat = basis2schmidt(psn_st)
    psn_dm = basis_mat * psn_dm * basis_mat.dag()
    schmidt_coeff = np.real((sc_mat.full())**2)
    schmidt_coeff = np.reshape(schmidt_coeff, [len(schmidt_coeff)])

    return schmidt_coeff, psn_dm, psn_st, [u_mat, v_mat]


def slocc_povm_on_dm_qobj(gamma_new_dm, povm_mat):
    """function for applying  povm to the gamma obtained in qboj form"""


    povm_mat = qt.Qobj(np.sqrt(povm_mat), dims = [[2,2,2], [2,2,2]])
    povm_mat = qt.tensor(I, povm_mat, I,I,I)

    out_dm = povm_mat* gamma_new_dm* povm_mat.dag()
    prob = 0
    #print(out_dm)
    prob = (out_dm).norm()
    if prob > 1.001:

        print("Error: prob greater than 1")

    assert prob != 0
    out_dm = out_dm/prob


    return out_dm, prob


def locc_povm_on_dm_qobj(input_state_dm, povm_out_list, perm_out_list):
    """gives out gamma in density matrix form with catalyst"""

    out_states = []
    out_probs = []
    out_state_dm = 0

    for i, element in enumerate(povm_out_list):

        element = qt.Qobj(np.sqrt(element), dims  = [[2,2,2], [2,2,2]])
        element = qt.tensor(I, element, I, I, I)
        perm_out_list[i] = qt.Qobj(perm_out_list[i], dims  = [[2,2,2], [2,2,2]])
        perm_out_list[i] = qt.tensor(I, perm_out_list[i], perm_out_list[i])


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

    povm_mat = qt.Qobj(np.sqrt(povm_mat), dims = [[2,2], [2,2]])
    povm_mat = qt.tensor(I, povm_mat, I,I)

    out_dm = povm_mat* gamma_new_dm* povm_mat.dag()
    prob = (out_dm).norm()
    assert prob != 0
    out_dm = out_dm/prob

    return out_dm, prob


def locc_povm_on_dm_qobj_nc(input_state_dm, povm_out_list, perm_out_list):
    """gives out gamma in density matrix form without catalyst"""
    out_states = []
    out_probs = []
    out_state_dm = 0

    for i, element in enumerate(povm_out_list):
        element = qt.Qobj(np.sqrt(element), dims  = [[2,2], [2,2]])

        element = qt.tensor(I, element, I, I)
        perm_out_list[i] = qt.Qobj(perm_out_list[i], dims  = [[2,2], [2,2]])
        perm_out_list[i] = qt.tensor(I, perm_out_list[i], perm_out_list[i])

        out_dm = element * input_state_dm * (element).dag()
        out_dm = perm_out_list[i].trans() * out_dm * perm_out_list[i]

        prob = out_dm.norm()

        out_state_dm += out_dm
        out_states.append(out_dm)
        out_probs.append(prob)

    return out_state_dm, out_probs



def prepare_carbon_spins(out_state, input_schmidt_coeff, psn_dm, pure_state):
    """find the optimal catalyst state and convert it into qt.Qobj form"""

    _, gain, _, cat_final, _ = catalytic_concentration(out_state, input_schmidt_coeff, 1, 2)
    #print(cat_final, "cat_final")
    carbon_st = (np.sqrt(cat_final[0])*qt.tensor(qt.basis(2,0), qt.basis(2,0))+
                 np.sqrt(cat_final[1])*qt.tensor(qt.basis(2,1), qt.basis(2,1)))

    carbon_dm = qt.ket2dm(carbon_st)
    #print(carbon_dm)

    psnc_dm = qt.tensor(carbon_dm, psn_dm)
    pure_state = qt.tensor(carbon_st, pure_state)
    #print(psn_dm)#, "pure-state")

    """putting the qt.tensor product in the form |ph>|sp_a>|nu_a>|c>_a|sp_b>|nu_b>|c_b>"""
    pure_state_cat = pure_state.permute([2,3,4,0,5,6,1])
    #print(pure_state_cat)
    psnc_dm = psnc_dm.permute([2,3,4,0,5,6,1])
    #print(cat_final, "cat_final")
    #print(psnc_dm)

    return carbon_st, psnc_dm, pure_state_cat, cat_final, gain


def cat_st2cat_array(catalyst_st):
    """converts catalyst state to array of schmidt coefficients"""
    #print(catalyst_st)
    catalyst_st = catalyst_st.full()
    cat_st_matrix_form = np.reshape(catalyst_st, (2, 2))
    _, s_mat, _ = np.linalg.svd(cat_st_matrix_form, full_matrices=True)
    s_coeff_cat = qt.Qobj(s_mat)

    return np.real((s_coeff_cat.full())**2)


def non_catalytic_conversion(prepared_dm):
    """function for taking in two prepared bell states and converts them
    by (1) finding the povms, and
    (2) applys the povms"""


    s_coeff, prepared_dm_schmidt_basis, _, _ = schmidt_decomp_of_dm(prepared_dm)
    input_state_array = np.reshape(np.real(s_coeff), [4])

    assert np.sum(input_state_array) != 0
    input_state_array = input_state_array/np.sum(input_state_array)


    output_state = [0.5, 0.5]
    output_state_array = concat_zeros(output_state, input_state_array)

    output_state_qobj = 1/np.sqrt(2)*(qt.tensor(zero,zero,zero,zero,zero) +
                                     qt.tensor(zero,zero,one,zero,one))


    """use function for gamma to find the ideal gamma that is needed"""
    gamma_ideal = func_for_gamma(output_state_array, input_state_array)
    assert np.sum(gamma_ideal) != 0
    gamma_ideal = gamma_ideal/np.sum(gamma_ideal)


    """use function for locc povms to find the ideal povms to get to gamma"""
    povm_out_list, _, perm_out_list = locc_povm_func(gamma_ideal, input_state_array)



    """use function for slocc povms to find the ideal povms to get to final state"""
    meas_mat, _ = slocc_povm_func(output_state_array, input_state_array)

    #print(input_state_array)
    #print(prepared_dm_schmidt_basis)
    """apply those povm on the density matrix"""
    gamma_density_mat, prob_obtained_list = locc_povm_on_dm_qobj_nc(
        prepared_dm_schmidt_basis, povm_out_list, perm_out_list)
    #print(gamma_density_mat.tidyup())
    assert len(prob_obtained_list) != 0
    """for gamma density matrix, apply the slocc povm to get output states"""
    probab = 0
    output_density_mat, probab = slocc_povm_on_dm_qobj_nc(gamma_density_mat, meas_mat)


    fid = qt.fidelity(output_state_qobj, output_density_mat)
    return fid, np.real(probab), output_density_mat, gamma_density_mat



def pre_conversion_process(prepared_dm):
    """function for creating the required arrays and qobjs
    for conversion of states"""

    s_coeff, prepared_dm, pure_st,_ = schmidt_decomp_of_dm(prepared_dm)
    s_coeff = np.reshape(s_coeff, [4])
    #print(s_coeff)

    """finding and making the catalyst for the pure statestate"""
    output_states = [0.5, 0.5]
    carbon_st, psnc_dm, psnc_st, cat_array, _ = prepare_carbon_spins(
        output_states, s_coeff, prepared_dm, pure_st)
    #print(cat_array)

    psnc_st, basis_mat_cat, ss_cat, u_mat, v_mat = basis2schmidt(psnc_st)
    psnc_dm = basis_mat_cat * psnc_dm * basis_mat_cat.dag()
    #print(psnc_st)
    #print(psnc_dm)
    input_state_array = np.reshape(np.real((ss_cat.full())**2), [8])
    output_state_array = np.sort(np.reshape(np.tensordot(output_states, cat_array, 0), 4))[::-1]
    output_state_array = concat_zeros(output_state_array, input_state_array)

    output_state_qobj = 1/np.sqrt(2)*(qt.tensor(zero,zero,zero,zero,zero) +
                                     qt.tensor(zero,zero,one,zero,one))
    output_state_qobj = qt.tensor(carbon_st, output_state_qobj)
    output_state_qobj = qt.ket2dm(output_state_qobj.permute([2,3,0,4,5,1,6]))

    return output_state_qobj, psnc_dm, carbon_st, input_state_array, output_state_array, [u_mat, v_mat]

def catalytic_conversion(prepared_dm):
    """function for taking in two prepared bell states and converts them
    by (1) finding the optimal catalyst, (2) finding the povms, and
    (3) applys the povms"""
    #print(prepared_dm)
    output_state_qobj, psnc_dm, carbon_st, input_state_array, output_state_array, _ = pre_conversion_process(prepared_dm)

    """use function for gamma to find the ideal gamma that is needed"""
    gamma_ideal = func_for_gamma(output_state_array, input_state_array)

    assert np.sum(gamma_ideal) != 0
    gamma_ideal = gamma_ideal/np.sum(gamma_ideal)

    majorisation_check(gamma_ideal, input_state_array)

    if np.any(input_state_array == 0):
        print(input_state_array, "input")
        print(gamma_ideal, "gamma")
        print(carbon_st, "carbon")
        print("input state array has a 0")
    """use function for locc povms to find the ideal povms to get to gamma"""
    povm_out_list, _, perm_out_list = locc_povm_func(gamma_ideal,
                                                                 input_state_array)
    #print(len(povm_out_list))
    """apply those povm on the density matrix"""
    gamma_density_mat, prob_obtained_list = locc_povm_on_dm_qobj(psnc_dm,
                                                                 povm_out_list, perm_out_list)

    assert len(prob_obtained_list) != 0

    """use function for slocc povms to find the ideal povms to get to final state"""
    meas_mat, _ = slocc_povm_func(output_state_array, input_state_array)

    """for each gamma density matrix, apply the slocc povm to get output states"""
    probab = 0
    output_density_mat, probab = slocc_povm_on_dm_qobj(gamma_density_mat, meas_mat)

    fid = qt.fidelity(output_state_qobj, output_density_mat)

    return fid, np.real(probab), output_density_mat, carbon_st



def catalytic_conversion_reuse(prepared_dm, carbon_dm_input):
    """function for taking in two prepared bell states and converts them
    by (1) finding the optimal catalyst, (2) finding the povms, and
    (3) applys the povms, (4) reusing the catalyst"""

    flagg = 0
    s_coeff, prepared_dm_schmidt_basis, pure_st, _ = schmidt_decomp_of_dm(prepared_dm)
    s_coeff = np.reshape(s_coeff, [4])

    """finding and making the catalyst for the pure statestate"""
    output_states = [0.5, 0.5]
    #carbon_st, psnc_dm, psnc_st, cat_array, cat_gain = prepare_carbon_spins(
    #    output_states, s_coeff, prepared_dm_schmidt_basis, pure_st)

    carbon_st, _ = closest_pure_state(carbon_dm_input)
    #print(qt.fidelity(carbon_dm_input, carbon_st))

    psnc_dm = qt.tensor(carbon_dm_input, prepared_dm_schmidt_basis)
    psnc_st = qt.tensor(carbon_st, pure_st)

    """putting the tensor product in the form |ph>|sp_a>|nu_a>|c>_a|sp_b>|nu_b>|c_b>"""
    psnc_st = psnc_st.permute([2,3,4,0,5,6,1])
    psnc_dm = psnc_dm.permute([2,3,4,0,5,6,1])

    cat_array = cat_st2cat_array(carbon_st)

    psnc_st, basis_mat_cat, ss_cat, _, _ = basis2schmidt(psnc_st)
    psnc_dm = basis_mat_cat * psnc_dm * basis_mat_cat.dag()

    input_state_array = np.reshape(np.real((ss_cat.full())**2), [8])
    output_state_array = np.sort(np.reshape(np.tensordot(output_states, cat_array, 0), 4))[::-1]
    output_state_array = concat_zeros(output_state_array, input_state_array)

    output_state_qobj = 1/np.sqrt(2)*(qt.tensor(zero,zero,zero,zero,zero) +
                                     qt.tensor(zero,zero,one,zero,one))
    output_state_qobj = qt.tensor(carbon_st, output_state_qobj)
    output_state_qobj = output_state_qobj.permute([2,3,0,4,5,1,6])

    """use function for gamma to find the ideal gamma that is needed"""
    gamma_ideal = func_for_gamma(output_state_array, input_state_array)
    assert np.sum(gamma_ideal) != 0
    gamma_ideal = gamma_ideal/np.sum(gamma_ideal)
    #print(cat_array)
    #print(input_state_array)
    if np.any(input_state_array <= 1e-12):
        flagg = 1
        output_density_mat = qt.tensor(carbon_dm_input, prepared_dm_schmidt_basis)
        output_density_mat = output_density_mat.permute([2,3,0,4,5,1,6])
        fid = 0
        probab = 0

        #print(carbon_st, "carbon")
        #print("state too small")

    else:
        """use function for locc povms to find the ideal povms to get to gamma"""
        povm_out_list, _, perm_out_list = locc_povm_func(gamma_ideal, input_state_array)

        """apply those povm on the density matrix"""
        gamma_density_mat, prob_obtained_list = locc_povm_on_dm_qobj(psnc_dm,
                                                povm_out_list, perm_out_list)

        assert len(prob_obtained_list) != 0

        """use function for slocc povms to find the ideal povms to get to final state"""
        meas_mat, _ = slocc_povm_func(output_state_array, input_state_array)

        """for each gamma density matrix, apply the slocc povm to get output states"""
        probab = 0
        output_density_mat, probab = slocc_povm_on_dm_qobj(gamma_density_mat, meas_mat)
        fid = qt.fidelity(output_state_qobj, output_density_mat)

    return fid, np.real(probab), output_density_mat, flagg
