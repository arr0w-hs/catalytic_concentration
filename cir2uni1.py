#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 12 20:14:50 2024

@author: hsharma4
"""

import sys
import os

import pickle
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

import qutip as qt
from qutip.measurement import measure, measurement_statistics, measure_observable
from qutip.qip.operations import expand_operator
from pathlib import Path
from base_state_transform import *
# from base_siv_state_prep import prepare_dm_withreset, l_vector, r_vector
from base_distillation import distillation, distillation_operation
from syn2depol_new import depol_final_state, measure_aux, to_be_syn_cat, to_be_syn_nc
from syn2depol import extend_perm
from base_depol_channels import new_state_pauli_x1#new_state_depol, new_state_pauli_z,
from bqskit.ir.circuit import Circuit
from qutip_qip.circuit import QubitCircuit
from base_locc_alt import locc_operations, slocc_operations

import numpy as np
import pandas as pd
import time
#import rsmf
#print(dir_name+"/quantum-template.tex")
# Get formatter specifications from tex file
#fmt = rsmf.setup(dir_name+"/quantum-template.tex")
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
    print(np.isclose(unitary.all(), np.idetity(num_qubits)))
    if np.isclose(unitary.all(), np.idetity(num_qubits)):
        
        return rho_in

    gates = [X, Z]
    for i, ele in enumerate(gates):
        if not perm_flag:
            for target_qubit in range(1, int(num_qubits/2)):
                print(target_qubit)
                gates[i] = expand_operator(ele, dims=[2]*num_qubits, targets=target_qubit)

        else:
            for target_qubit in range(1, num_qubits):
                gates[i] = expand_operator(ele, dims=[2]*num_qubits, targets=target_qubit)
                print(target_qubit)

    rho_out = (1-err_prob)*rho_in + err_prob/3*(gates[0]*rho_in*gates[0] +
                                                gates[0]*gates[1]*rho_in*gates[1]*gates[0] +
                                                gates[1]*rho_in*gates[1])

    return rho_out


def povm_perm_one_round(prepared_dm, locc_povm, locc_perm_list, cat_flag, error_rate):
    psn_aux_dm = add_aux(prepared_dm, cat_flag)

    num_qubits = int(6+2*cat_flag)

    for unitary in locc_povm:
        unitary = qt.tensor(I, unitary,I, I,I)
        if cat_flag:
            unitary = qt.tensor(unitary, I, I)
        psn_aux_dm = unitary*psn_aux_dm*unitary.dag()
        psn_aux_dm = depol_channel(psn_aux_dm, error_rate, unitary)

    mea_list = measure_aux(psn_aux_dm, num_qubits, cat_flag)
    meas_perm = list(zip(mea_list, locc_perm_list))
    os = 0
    for out_state, perm in meas_perm:
        perm = qt.tensor(I, perm,I,perm)
        # if cat_flag:
        #     perm = qt.tensor(I, perm,I,perm)
        out_state = perm*out_state*perm.dag()
        out_state = depol_channel(out_state, error_rate, perm, perm_flag=1)
        os += out_state

    # out_state = os0+os1

    if cat_flag == 1:
        out_state = out_state.ptrace([0,1,2,3,5,6,7])
    else:
        #print(os.ptrace([3]), "partial trace")
        out_state = out_state.ptrace([0,1,2,4,5])
    #print(" ")
    return out_state



def apply_povm_n_perm(povm_circuit, permu_circ_list, prepared_dm, cat_flag, sq_error_rate, cnot_error_rate):

    psn_aux_dm = add_aux(prepared_dm, cat_flag)

    assert cat_flag == 0 or cat_flag == 1
    #print(" ")
    num_qubits = int(6+2*cat_flag)

    out_state = depol_final_state(povm_circuit, psn_aux_dm, sq_error_rate, cnot_error_rate, num_qubits, 0, cat_flag, 0)
    #os = depolo(syn_circ, in_state, sq_error_rate, cnot_error_rate, num_qubits, bob_flag, cat_flag)

    mea_dm0, mea_dm1 = measure_aux(out_state, num_qubits, cat_flag)

    #print(qt.Qobj(permu_circ_list[0].get_unitary()).tidyup())
    #print(permu_circ_list[1].get_unitary())
    #applying permutations
    os0 = depol_final_state(permu_circ_list[0], mea_dm0, sq_error_rate, cnot_error_rate, num_qubits, 0, cat_flag, 1)
    os0 = depol_final_state(permu_circ_list[0], os0, sq_error_rate, cnot_error_rate, num_qubits, 1, cat_flag, 1)
    os1 = depol_final_state(permu_circ_list[1], mea_dm1, sq_error_rate, cnot_error_rate, num_qubits, 0, cat_flag, 1)
    os1 = depol_final_state(permu_circ_list[1], os1, sq_error_rate, cnot_error_rate, num_qubits, 1, cat_flag, 1)



    out_state = os0+os1



    if cat_flag == 1:
        out_state = out_state.ptrace([0,1,2,3,5,6,7])
    else:
        #print(os.ptrace([3]), "partial trace")
        out_state = out_state.ptrace([0,1,2,4,5])
    #print(" ")
    return out_state


def apply_locc_conversion(locc_oper_list, prepared_dm, cat_flag, sq_error_rate, cnot_error_rate):

    num_comm_rounds = len(locc_oper_list)
    for i in range(num_comm_rounds):
        #print(i, "i")
        povm_circ = locc_oper_list[i][0]
        permu_circ_list = locc_oper_list[i][1]
        # out_state = apply_povm_n_perm(povm_circ, permu_circ_list, prepared_dm, cat_flag, sq_error_rate, cnot_error_rate)
        out_state = povm_perm_one_round(prepared_dm, povm_circ, permu_circ_list, cat_flag, sq_error_rate)
        prepared_dm = out_state

    return out_state

def apply_slocc_conversion(slocc_circ, prepared_dm, cat_flag, sq_error_rate, cnot_error_rate):

    assert cat_flag == 0 or cat_flag == 1

    psn_aux_dm = add_aux(prepared_dm, cat_flag)
    num_qubits = int(6+2*cat_flag)
    out_state = depol_final_state(slocc_circ, psn_aux_dm, sq_error_rate, cnot_error_rate, num_qubits, 0, cat_flag, 1)

    mea_dm0, mea_dm1 = measure_aux(out_state, num_qubits, cat_flag)
    prob0 = mea_dm0.norm()
    mea_dm0 = mea_dm0.unit()

    if cat_flag == 1:
        mea_dm0 = mea_dm0.ptrace([0,1,2,3,5,6,7])
    else:
        #print(mea_dm0.ptrace([3]), "partial trace")
        mea_dm0 = mea_dm0.ptrace([0,1,2,4,5])

    return mea_dm0, prob0

def apply_schmidt_conversion(u_circ, v_circ, in_state, cat_flag, sq_error_rate, cnot_error_rate):

    assert cat_flag == 0 or cat_flag == 1

    num_qubits = int(5+2*cat_flag)

    #depol_final_state(syn_circ, in_state, sq_error_rate, cnot_error_rate, num_qubits, bob_flag, cat_flag)
    out_state = depol_final_state(u_circ, in_state, sq_error_rate, cnot_error_rate, num_qubits, 0, cat_flag, 0)
    out_state = depol_final_state(v_circ, out_state, sq_error_rate, cnot_error_rate, num_qubits, 1, cat_flag, 0)

    return out_state


def oper_err(list_circs, prepared_state, cat_flag, sqe_error, cnot_error):

    zero = basis(2,0)
    one = basis(2,1)
    output_state_qobj = 1/np.sqrt(2)*(tensor(zero,zero,zero,zero,zero) +
                                     tensor(zero,zero,one,zero,one))

    locc_op = list_circs[0]
    slocc_op = list_circs[1]
    u_op = list_circs[2]
    v_op = list_circs[3]

    if cat_flag ==1:

        s_coeff, prepared_dm_schmidt_basis, pure_st,_ = schmidt_decomp_of_dm(prepared_state)
        s_coeff = np.reshape(s_coeff, [4])
        #print(s_coeff)
        """finding and making the catalyst for the pure statestate"""
        output_states = [0.5, 0.5]
        carbon_st, prepared_state, psnc_st, cat_array, cat_gain = prepare_carbon_spins(
            output_states, s_coeff, prepared_state, pure_st)

        output_state_qobj = tensor(carbon_st, output_state_qobj)
        output_state_qobj = ket2dm(output_state_qobj.permute([2,3,0,4,5,1,6]))

    final_state_sd = apply_schmidt_conversion(u_op, v_op, prepared_state, cat_flag, sqe_error, cnot_error)
    gamma_got = apply_locc_conversion(locc_op, final_state_sd, cat_flag, sqe_error, cnot_error)
    output_state, prob = apply_slocc_conversion(slocc_op, gamma_got, cat_flag, sqe_error, cnot_error)

    fid = fidelity(output_state, output_state_qobj)

    return fid, prob, output_state


if __name__ == "__main__":
    a = 1
    p = 1
    final_state = new_state_pauli_x1(a, p)
    ideal_state = new_state_pauli_x1(1, 1)

    # out = depol_channel(ideal_state, 0.5)
    # print(out)
    out_state = apply_locc_conversion(final_state, 1, 0, 0)

    output_state_qobj = 1/np.sqrt(2)*(qt.tensor(zero,zero,zero,zero,zero) +
                                     qt.tensor(zero,zero,one,zero,one))
    output_state_qobj = qt.tensor(carbon_st, output_state_qobj)
    output_state_qobj = output_state_qobj.permute([2,3,0,4,5,1,6])

    print(qt.fidelity(out_state, output_state_qobj))
    # fid_dist, prob, ops = distillation(final_state, 0)
    # print(fid_dist, prob, "distillation")
    # print(ops)
    # fid_nocat, prob_nocat, post_locc_state_nocat = oper_err(list_circ_nc, final_state, 0, 0, 0)
    # print(fid_nocat, prob_nocat, "slocc")
    
    # fid_cat, prob_cat, post_locc_state_cat = oper_err(cat, final_state, 1, 0, 0)
    # print(fid_cat, prob_cat, "cat_slocc")

    # print(hh)
    
    # x = []
    # fid_list = []
    # prob_list = []
    
    # fid_cat_list = []
    # prob_cat_list = []
    
    # fid_dist_list = []
    # prob_dist_list = []
    
    
    # #print(output_state_qobj)
    # for i in range(0):
    #     print(i)
    #     t1 = time.time()
    #     err = (i)*0.001
    #     cerr = err
    #     x.append(err)
    #     fid_nocat, prob_nocat, post_locc_state_nocat = oper_err(list_circ_nc, final_state, 0, err, cerr)
    
    #     hh = distillation_operation(final_state, ops, err, cerr)
    
    
    #     fid_dist_list.append(hh[0])
    #     prob_dist_list.append(hh[1])
    
    #     fid_list.append(fid_nocat)
    #     prob_list.append(prob_nocat)
    
    #     fid_cat, prob_cat, post_locc_state_cat = oper_err(cat, final_state, 1, err, cerr)
    #     fid_cat_list.append(fid_cat)
    #     prob_cat_list.append(prob_cat)
    
    #     #print(i, time.time()-t1)
    
    
    # #fig = fmt.figure()
    # plt.figure()
    # plt.plot(x, fid_cat_list, label = "CEC")
    # plt.plot(x, fid_list, label = "SEC")
    # plt.plot(x, fid_dist_list, label = "Distillation")
    # #plt.yscale("log")
    # plt.legend()
    # plt.ylabel('Fidelity')
    # plt.xlabel('Error rate')
    # plt.xticks(rotation=45)
    # plt.grid()
    # # plt.savefig(dir_name+"/_fidelity_cat_sqe1.png", dpi=1000, format="png", bbox_inches = 'tight')
    # # plt.savefig(dir_name+"/_fidelity_cat_sqe1.svg", dpi=1000, format="svg", bbox_inches = 'tight')
    # #plt.show()
    
    # plt.figure()
    # plt.grid()
    # plt.plot(x, prob_cat_list, label = "CEC")
    # plt.plot(x, prob_list, label = "SEC")
    # plt.plot(x, prob_dist_list, label = "Distillation")
    # #plt.yscale("log")
    # plt.legend()
    # plt.ylabel('Probability of success')
    # plt.xlabel('Error rate')
    # plt.xticks(rotation=45)
    # # plt.savefig(dir_name+"/_probability_cat_sqe1" + ".png", dpi=1000, format="png", bbox_inches = 'tight')
    # # plt.savefig(dir_name+"/_probability_cat_sqe1" + ".svg", dpi=1000, format="svg", bbox_inches = 'tight')
    # plt.show()
