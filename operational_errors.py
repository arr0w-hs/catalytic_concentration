#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 07:24:57 2025

@author: hsharma4
"""


import sys
import os
#from pathlib import Path
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

import qutip as qt
#from qutip.measurement import measure, measurement_statistics, measure_observable
from qutip.qip.operations import expand_operator
from base_state_transform import schmidt_decomp_of_dm, pre_conversion_process
# from base_siv_state_prep import prepare_dm_withreset, l_vector, r_vector
from base_distillation import distillation, distillation_operation
from syn2depol import extend_perm
from base_slocc import concat_zeros, func_for_gamma, slocc_povm_func
from base_depol_channels import new_state_pauli_x1#new_state_depol, new_state_pauli_z,
from base_locc_alt import locc_operations, slocc_operations
from base_transform import catalytic_conversion, basis2schmidt
from base_transform import non_catalytic_conversion
from base_transform import prepare_carbon_spins

import numpy as np
import time

plt.rcParams.update({'font.size': 12})

zero = qt.basis(2,0)
one = qt.basis(2,1)
I = qt.qeye(2)
X = qt.sigmax()
Z = qt.sigmaz()
Y = qt.sigmay()
H = 1/np.sqrt(2)*(X+Z)

def add_aux(prepared_dm, cat_flag):


    aux = qt.ket2dm(qt.basis(2,0))
    if cat_flag == 1:
        psn_aux_dm = qt.tensor(prepared_dm, aux).permute([0,1,2,3,7,4,5,6])
    else:
        psn_aux_dm = qt.tensor(prepared_dm, aux).permute([0,1,2,5,3,4])

    return psn_aux_dm


def measure_aux(input_dm, num_qubits, cat_flag):

    """aux is at the (4+c)th location that is the (3+c)th index in the qobj

    returns: the output states after the two measurements"""
    dimens = [2]*num_qubits
    zero = qt.ket2dm(qt.basis(2,0))
    one_1 = X*qt.ket2dm(qt.basis(2,1)) #X is for taking the aux qubit to 0 after measurement

    mea0 = expand_operator(zero, dims=dimens, targets=[int(2+cat_flag)])
    mea1 = expand_operator(one_1, dims=dimens, targets=[int(2+cat_flag)])

    mea0_out = mea0*input_dm*mea0.dag()
    mea1_out = mea1*input_dm*mea1.dag()

    return mea0_out, mea1_out


def line_format(perm):
    perm_dim = int(np.log2(np.shape(perm)[0]))
    line_form = [[j for j, elem in enumerate(ele) if elem == 1] for i, ele in enumerate(perm)]
    line_form = [ele[0] for ele in line_form]

    line_form = [ele for i, ele in enumerate(line_form) if i!= ele]


    if len(line_form) == 0:
        return 0
    else:
        line_form = [np.binary_repr(ele, width=perm_dim) for ele in line_form]
        line_form = list(zip((line_form[0]), (line_form[1])))

        hamming_dist = [(int(a)+int(b))%2 for a,b in line_form]
        hamming_dist = sum(hamming_dist)

        locations = [a+b for i, (a, b) in enumerate(list(line_form))]
        return hamming_dist


# def find_permutation(line_form):



    return

def extend_perm(perm_list, num_qubits):
    """function for extending the dimensions of permutation unitaries
       to dimensions of the initial state

       inputs:
           1. list of permutations
           2. initial state density matrix with only alices qubits

       returns: list of extended permutation unitaries
    """

    required_len = int(2**num_qubits)
    initial_dims = [[2]*num_qubits, [2]*num_qubits]

    extended_perm_list = []
    for i in range(len(perm_list)):

        perm_mat = perm_list[i]
        perm_len = np.shape(perm_mat)[0]

        extra_len = int(required_len-perm_len)

        identity_temp = np.eye(extra_len)

        side_zeros = np.zeros((perm_len, extra_len))
        lower_zeros = np.zeros((extra_len, perm_len))

        extended_perm_mat = np.block([[perm_mat, side_zeros],
                  [lower_zeros, identity_temp]
                  ])

        extended_perm_mat = qt.Qobj(extended_perm_mat, dims = initial_dims)
        extended_perm_list.append(extended_perm_mat)


    return extended_perm_list


def depol_channel(rho_in, err_prob, unitary, perm_flag=0):
    """
    function for depol channel at one qubit

    inputs:
        input dm
        errors rates
        location of the qubit
        total num of qubits at alice's side (including ancilla)
    returns:
        output dm with depol channel on one qubit
    """
    num_qubits = int(np.ceil(np.log2(rho_in.shape[0])))
    if np.array_equiv(unitary.full(), np.identity(2**num_qubits)):

        return rho_in

    gates = [X, Z]
    for i, ele in enumerate(gates):
        if not perm_flag:

            for target_qubit in range(0, int(num_qubits/2)):
                gates[i] = expand_operator(ele, dims=[2]*num_qubits, targets=target_qubit)

        else:
            for target_qubit in range(0, num_qubits):
                gates[i] = expand_operator(ele, dims=[2]*num_qubits, targets=target_qubit)

    rho_out = (1-err_prob)*rho_in + err_prob/3*(gates[0]*rho_in*gates[0] +
                                                gates[0]*gates[1]*rho_in*gates[1]*gates[0] +
                                                gates[1]*rho_in*gates[1])

    return rho_out


def povm_perm_one_round(prepared_dm, locc_povm, locc_perm_list, cat_flag, error_rate):

    psn_aux_dm = add_aux(prepared_dm, cat_flag)
    if cat_flag == 1:
        psn_aux_dm = psn_aux_dm.ptrace([1,2,3,4,5,6,7])
    else:
        psn_aux_dm = psn_aux_dm.ptrace([1,2,3,4,5])
    num_qubits = int(5+2*cat_flag)

    for unitary in locc_povm:
        unitary = qt.tensor(unitary,I, I)
        if cat_flag:
            unitary = qt.tensor(unitary, I)

        psn_aux_dm = unitary*psn_aux_dm*unitary.dag()
        psn_aux_dm = depol_channel(psn_aux_dm, error_rate, unitary)
        psn_aux_dm = depol_channel(psn_aux_dm, error_rate, unitary)

    mea_list = measure_aux(psn_aux_dm, num_qubits, cat_flag)
    locc_perm_list = extend_perm(locc_perm_list, int(num_qubits/2))

    meas_perm = list(zip(mea_list, locc_perm_list))
    os = 0

    for os_temp, perm in meas_perm:

        if cat_flag == 1:
            os_temp = os_temp.ptrace([0,1,2,4,5,6])
        else:
            os_temp = os_temp.ptrace([0,1,3,4])

        perm_len = np.shape(perm)[0]

        num_qubits = int(np.ceil(np.log2(perm_len)))

        perm = qt.Qobj(perm, dims = [[2]*num_qubits]*2)
        perm = qt.tensor(perm,perm)

        os_temp = perm*os_temp*perm.dag()
        # os_temp = depol_channel(os_temp, error_rate, perm, perm_flag=1)

        os += os_temp

    out_state = os
    out_state = qt.tensor(qt.ket2dm(zero), out_state)

    return out_state



def to_be_syn_nc(prepared_dm):

    """pre_conv_process gives out the u and v matrices in catalytic conversion"""

    s_coeff, prepared_dm_schmidt_basis, pure_st, basis_mat = schmidt_decomp_of_dm(prepared_dm)
    input_state_array = np.reshape(np.real(s_coeff), [4])

    output_state = [0.5, 0.5]
    output_state_array = concat_zeros(output_state, input_state_array)

    """use function for gamma to find the ideal gamma that is needed"""
    gamma_ideal = func_for_gamma(output_state_array, input_state_array)

    """use function for locc povms to find the ideal povms to get to gamma"""
    operations_list = locc_operations(gamma_ideal, input_state_array)

    """use function for slocc povms to find the ideal povms to get to final state"""
    slocc_list = slocc_povm_func(output_state_array, input_state_array)
    slocc_uni = slocc_operations(slocc_list)

    return operations_list, slocc_uni, basis_mat

def to_be_syn_cat(prepared_dm):

    """pre_conv_process gives out the u and v matrices in catalytic conversion"""
    output_state_qobj, psnc_dm, carbon_st, input_state_array, output_state_array, basis_mat = pre_conversion_process(prepared_dm)

    """use function for gamma to find the ideal gamma that is needed"""
    gamma_ideal = func_for_gamma(output_state_array, input_state_array)

    """use function for locc povms to find the ideal povms to get to gamma"""
    operations_list = locc_operations(gamma_ideal, input_state_array)

    """use function for slocc povms to find the ideal povms to get to final state"""
    slocc_povm = slocc_povm_func(output_state_array, input_state_array)
    slocc_uni = slocc_operations(slocc_povm)


    return operations_list, slocc_uni, basis_mat


def apply_locc_conversion(locc_oper_list, prepared_dm, cat_flag, error_rate):

    num_comm_rounds = len(locc_oper_list)
    out_state = prepared_dm
    for i in range(num_comm_rounds):

        povm_list = locc_oper_list[i][0]
        perm_list = locc_oper_list[i][1]
        # out_state = apply_povm_n_perm(povm_circ, permu_circ_list, prepared_dm, cat_flag, error_rate, cnot_error_rate)
        out_state = povm_perm_one_round(prepared_dm, povm_list, perm_list, cat_flag, error_rate)
        prepared_dm = out_state

    return out_state


def apply_slocc_conversion(slocc_uni_list, prepared_dm, cat_flag, error_rate):

    num_qubits = int(2+cat_flag)

    psn_aux_dm = add_aux(prepared_dm, cat_flag)
    if cat_flag == 1:
        psn_aux_dm = psn_aux_dm.ptrace([1,2,3,4,5,6,7])
    else:
        psn_aux_dm = psn_aux_dm.ptrace([1,2,3,4,5])
    num_qubits = int(5+2*cat_flag)

    for unitary in slocc_uni_list:
        unitary = qt.tensor(unitary,I, I)
        if cat_flag:
            unitary = qt.tensor(unitary, I)

        psn_aux_dm = unitary*psn_aux_dm*unitary.dag()
        psn_aux_dm = depol_channel(psn_aux_dm, error_rate, unitary)
        psn_aux_dm = depol_channel(psn_aux_dm, error_rate, unitary)

    mea_dm0, mea_dm1 = measure_aux(psn_aux_dm, num_qubits, cat_flag)
    prob0 = mea_dm0.norm()
    mea_dm0 = mea_dm0.unit()

    if cat_flag == 1:
        mea_dm0 = mea_dm0.ptrace([0,1,2,4,5,6])
    else:
        mea_dm0 = mea_dm0.ptrace([0,1,3,4])
    mea_dm0 = qt.tensor(I, mea_dm0)

    return mea_dm0, prob0


def apply_schmidt_conversion(unitary, in_dm, error_rate):

    in_dm = unitary*in_dm*unitary.dag()
    out_state = depol_channel(in_dm, error_rate, unitary, perm_flag=0)

    return out_state


def oper_err(prepared_state, cat_flag, error_rate):

    zero = qt.basis(2,0)
    one = qt.basis(2,1)
    output_state_qobj = 1/np.sqrt(2)*(qt.tensor(zero,zero,zero,zero,zero) +
                                     qt.tensor(zero,zero,one,zero,one))
    psnc_dm = 1
    if cat_flag == 1:

        s_coeff, prepared_state, pure_st,_ = schmidt_decomp_of_dm(prepared_state)
        s_coeff = np.reshape(s_coeff, [4])

        """finding and making the catalyst for the pure statestate"""
        output_states = [0.5, 0.5]
        carbon_st, psnc_dm, psnc_st, cat_array, _ = prepare_carbon_spins(
            output_states, s_coeff, prepared_state, pure_st)
        psnc_st, basis_mat, ss_cat, u_mat, v_mat = basis2schmidt(psnc_st)

        output_state_qobj = qt.tensor(carbon_st, output_state_qobj)
        output_state_qobj = qt.ket2dm(output_state_qobj.permute([2,3,0,4,5,1,6]))
        locc_op, slocc_uni, _ = to_be_syn_cat(prepared_state)
        prepared_state = psnc_dm
    else:
        locc_op, slocc_uni, basis_mat = to_be_syn_nc(prepared_state)

    final_state_sd = apply_schmidt_conversion(basis_mat, prepared_state, error_rate)
    gamma_got = apply_locc_conversion(locc_op, final_state_sd, cat_flag, error_rate)
    output_state, prob = apply_slocc_conversion(slocc_uni, gamma_got, cat_flag, error_rate)
    fid = qt.fidelity(output_state, output_state_qobj)

    return fid, prob, output_state, psnc_dm


if __name__ == "__main__":


    a = 0.85
    p = 0.95

    final_state = new_state_pauli_x1(a, p)
    # ideal_state = new_state_pauli_x1(1, 1)
    # final_state = final_state.ptrace([1,2,3,4])
    # out = depol_channel(ideal_state, 0.5)
    # print(out)
    out_state = oper_err(final_state, 1, 0)


    aa = catalytic_conversion(final_state)
    # print(qt.fidelity(out_state[3], aa[4]))

    fid_cat, prob_cat, post_locc_state, carbon_cat_st = catalytic_conversion(final_state)
    print(fid_cat, prob_cat, "fid cat ")
    print()
    fid_nocat, prob_nocat, post_locc_state_nocat, _ = non_catalytic_conversion(final_state)#, prob_in_state, alpha)
    out_state1 = oper_err(final_state, 0, 0)
    print(fid_nocat, prob_nocat, "fid no cat")

    # fid_dist, prob_dist, _ = distillation(final_state, 0)
    # fid_dames, prob_dames = dejmps(final_state, 0)
    # print( fid_dist, prob_dist, "dist")#fid_dames, prob_dames, "dames",



    fid_dist, prob, ops = distillation(final_state, 0)
    print(fid_dist, prob, "distillation")
    print(ops)

    x = []
    fid_list = []
    prob_list = []

    fid_cat_list = []
    prob_cat_list = []

    fid_dist_list = []
    prob_dist_list = []


    # #print(output_state_qobj)
    for i in range(10):
        # print(i)
        t1 = time.time()
        # if i <10:
        #     err = 0.0001*i
        # elif i<20 and i>=10:
        #     err = 0.001*(i-10)
        # else:
        #     err = 0.01*(i-20)

        err = 1.2**(i)*0.0001

        # err = (400*(i+1))*0.00000005
        # print(i,err)
        # err = (i+1)*0.01
        cerr = err
        x.append(err)
        out_state = oper_err(final_state, 0, err)

        hh = distillation_operation(final_state, ops, err, cerr)


        fid_dist_list.append(1-hh[0])
        prob_dist_list.append(hh[1])

        fid_list.append(1-out_state[0])
        prob_list.append(out_state[1])

        out_state = oper_err(final_state, 1, err)
        fid_cat_list.append(1-out_state[0])
        prob_cat_list.append(out_state[1])

        #print(i, time.time()-t1)

    fs = 15
    #fig = fmt.figure()
    plt.figure()
    plt.plot(x, fid_cat_list, 'o-', label = "CEC")
    plt.plot(x, fid_list, '.-', label = "SEC")
    plt.plot(x, fid_dist_list, 'v-', label = "Distillation")
    # plt.yscale("log")
    # plt.xscale("log")
    # plt.legend()
    plt.ylabel('Infidelity', fontsize=fs)
    plt.xlabel('Error rate', fontsize=fs)
    plt.xticks(rotation=45, fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.legend(fontsize = fs,
               handlelength=1.3, handleheight=0.5, labelspacing = 0.15)
    plt.grid()
    # plt.savefig(dir_name+"/_fidelity_cat_oper_big.pdf", dpi=1000, format="pdf", bbox_inches = 'tight')
    # plt.savefig(dir_name+"/_fidelity_cat_oper_big.svg", dpi=1000, format="svg", bbox_inches = 'tight')




    #plt.show()

    plt.figure()
    plt.grid()
    plt.plot(x, prob_cat_list, 'o-', label = "CEC")
    plt.plot(x, prob_list, '.-', label = "SEC")
    plt.plot(x, prob_dist_list, 'v-', label = "Distillation")
    # plt.yscale("log")
    plt.xscale("log")
    # plt.legend()
    plt.ylabel('Probability of success', fontsize=fs)
    plt.xlabel('Error rate', fontsize=fs)
    plt.xticks(rotation=45, fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.legend(fontsize = fs,
               handlelength=1.3, handleheight=0.5, labelspacing = 0.15)
    # plt.savefig(dir_name+"/_probability_cat_oper_big" + ".pdf", dpi=1000, format="pdf", bbox_inches = 'tight')
    # plt.savefig(dir_name+"/_probability_cat_oper_big" + ".svg", dpi=1000, format="svg", bbox_inches = 'tight')
    plt.show()
