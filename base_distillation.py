#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 29 10:51:19 2023

@author: hsharma4
"""

import sys
import os

import numpy as np
import qutip as qt

from syn2depol import depol_channel
sys.path.append(os.path.dirname(__file__))

zero = qt.basis(2,0)
one = qt.basis(2,1)
I = qt.qeye(2)
X = qt.sigmax()
Z = qt.sigmaz()
Y = qt.sigmay()
H = 1/np.sqrt(2)*(X+Z)

ketbra0 = qt.ket2dm(qt.basis(2, 0))
ketbra1 = qt.ket2dm(qt.basis(2, 1))
plus = qt.ket2dm((qt.basis(2, 0)+qt.basis(2, 1)).unit())
minus = qt.ket2dm((qt.basis(2, 0)-qt.basis(2, 1)).unit())

S = ketbra0 + 1j*ketbra1

def err_cenotn(theta):
    """errr cenotn"""

    cnot = qt.tensor(ketbra0, I)+qt.tensor(ketbra1, X)
    cnot_bar = qt.tensor(ketbra0, X)+qt.tensor(ketbra1, I)
    #print(cnot_bar)
    err_cnot =  (np.cos(theta)*qt.tensor(I,I) - 1j*np.sin(theta)*cnot_bar)*cnot

    return err_cnot

def err_cnnote(theta):
    """errr cnnote"""

    cnot = ([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])
    cnot = qt.Qobj(cnot, dims = [[2,2],[2,2]])
    cnot_bar = ([[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
    cnot_bar = qt.Qobj(cnot_bar, dims = [[2,2],[2,2]])
    err_cnot =  (np.cos(theta)*qt.tensor(I,I) - 1j*np.sin(theta)*cnot_bar)*cnot

    return err_cnot

def swap_en(theta_en, theta_ne):
    """function for swapping erroneous swaps with different cnots"""

    swap = err_cenotn(theta_en)*err_cnnote(theta_ne)*err_cenotn(theta_en)

    return swap


def swap_en_a(cnot_error_en, cnot_error_ne):
    """
    swap en a and b are for swapping the states of electron and nu
    at alice and bobs nodes

    --input--
    the cnot fidelities of e and n as controls

    --returns--
    swap operator for alice and bob
    """

    swap_a = qt.tensor(I,swap_en(cnot_error_en, cnot_error_ne),I,I)

    return swap_a

def swap_en_b(cnot_error_en, cnot_error_ne):
    """bob's version of swap error"""
    swap_b = qt.tensor(I,I,I,swap_en(cnot_error_en, cnot_error_ne))

    return swap_b



def permutation_unitary(permutation_num):
    """for a given permutation number from 0 to 5, returns the permutation for alice and bob"""

    assert permutation_num <= 5 #("wrong permutation number")

    perm_mat_alice = I
    perm_mat_bob = I
    for i in range(permutation_num):
        perm_mat_alice = ((i+1)%2)*(H*perm_mat_alice)+(i%2)*(perm_mat_alice*S*H)
        perm_mat_bob = ((i+1)%2)*(H*perm_mat_bob)+(i%2)*(perm_mat_bob*H*S*H*S)

    return perm_mat_alice, perm_mat_bob

def permutation_distillation(psn, permutation_num, qubit):
    """if qubit is 0, bell state in spin is permuted
       if qubit is 1, bell state in nu is permuted"""

    perm_alice, perm_bob = permutation_unitary(permutation_num)

    perm_mat_alice = ((qubit%2)*qt.tensor(I, perm_alice, I,I,I)
                      +((qubit+1)%2)*qt.tensor(I,I, perm_alice, I,I))

    perm_mat_bob = ((qubit%2)*qt.tensor(I,I,I, perm_bob, I)
                    +((qubit+1)%2)*qt.tensor(I,I,I,I, perm_bob))

    psn = perm_mat_alice*perm_mat_bob*psn*perm_mat_alice.dag()*perm_mat_bob.dag()

    return psn

def coinc_distillation_final(psn, mea_basis, control, err_rate):

    """psn is the psn density matrix
       person is alice or bob
       basis is z (0), x (1) or y (2) bases
       target can be spin (0) or nu (1)"""

    if control == 1:
        psn = swap_en_b(err_rate, err_rate)*psn*swap_en_b(err_rate, err_rate).dag()
    #elif control ==2:
    #    psn = swap_en_a(err_rate, err_rate)*psn*swap_en_a(err_rate, err_rate).dag()
    #else:
    #    psn = swap_en_a(err_rate, err_rate)*psn*swap_en_a(err_rate, err_rate).dag()
    #    psn = swap_en_b(err_rate, err_rate)*psn*swap_en_b(err_rate, err_rate).dag()

    bell_st = 1/np.sqrt(2)*(qt.tensor(zero,zero,zero,zero,zero) +
                                     qt.tensor(zero,one,zero,one,zero))

    if mea_basis == 1:
        psn = qt.tensor(I,I,H,I,H)*psn*qt.tensor(I,I,H,I,H).dag()
    elif mea_basis == 2:
        psn = qt.tensor(I,I,S*H,I,S*H*X).dag()*psn*qt.tensor(I,I,S*H,I,S*H*X)

    meas0 = qt.ket2dm(zero)
    meas1 = zero*one.dag()

    meas0 = qt.tensor(I, I, meas0, I, meas0)
    meas1 = qt.tensor(I, I, meas1, I, meas1)

    psn_final = meas0*psn*meas0.dag()+meas1*psn*meas1.dag()
    prob = psn_final.tr()
    psn_final = psn_final.unit()

    fid_dist = qt.fidelity(bell_st, psn_final)

    return psn_final, np.real(fid_dist), prob

"""if control == 00, then alice and bob spin is the control
   if control == 01, then alice spin and bob nu is control
   if control == 10, then alice nu and bob spin is control
   if control == 11, then alice and bob nu is control"""
def cnot_distillation(psn, err_rate, control):
    """cnots during distillation"""

    cnot_mat = ((control%2)*err_cenotn(err_rate) +
                ((control+1)%2)*err_cnnote(err_rate))

    if control == 0:
        cnot_mat_a = err_cenotn(err_rate)
        cnot_mat_b = err_cenotn(err_rate)
    elif control == 1:
        cnot_mat_a = err_cenotn(err_rate)
        cnot_mat_b = err_cnnote(err_rate)

    cnot_mat = qt.tensor(I,cnot_mat_a,cnot_mat_b)

    psn = cnot_mat*psn*cnot_mat.dag()
    return psn

def distillation(prepared_state, cnot_err):
    """function for distillation"""
    fid_list = []
    prob_list = []
    fip_list = []


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
                _, fid_disti, prob_disti = coinc_distillation_final(psn_post_cnot, meas_basis, cnot_control, cnot_err)


                op_list.append([op_list1[i], meas_basis, cnot_control])
                fid_list.append(fid_disti)
                prob_list.append(prob_disti)
                fip_list.append(fid_disti*prob_disti)

    psn_list1.clear()
    perf_psn_list1.clear()
    op_list1.clear()

    maxarg1 = np.argmax(fid_list)
    fidy = fid_list[maxarg1]
    oper = op_list[maxarg1]

    return fidy, prob_list[maxarg1], oper

def distillation_operation(prepared_state, operation_list, sqe_err, cnot_err):
    """function for operational errors during distillation"""
    sqe_list = operation_list[0]
    meas = operation_list[1]
    cnot_control = operation_list[2]

    num_qubits = int(np.log2(prepared_state.shape[0]))


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

    _, fid_disti, prob_disti = coinc_distillation_final(psn_post_cnot, meas, cnot_control, 0)

    return fid_disti, prob_disti
