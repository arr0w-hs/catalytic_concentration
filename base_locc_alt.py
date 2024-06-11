#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:29:38 2024

@author: hsharma4

code for implementing Neilsen's POVM construction for LOCC transformations'
"""
import sys
import os
import numpy as np
#import scipy as sc
from qutip import *
#from qutip.measurement import measure, measurement_statistics, measure_observable

sys.path.append(os.path.dirname(__file__))
from base_slocc import *
from base_locc import *
from base_state_change import *

def create_ds_tlist(final_state, input_state):
    """
    output_state is the final state after LOCC transformation
    input state is the inital state for LOCC
    the method of construction of D-matrix goes from
    output state to input state

    this function gives out a list of t_matrices required for
    locc conversions during separate rounds
    """

    final_state = concat_zeros(final_state, input_state)

    count = 0
    assert (len(final_state) <=  len(input_state))
        #raise Exception("Incoherent dimensions of states")

    ips_temp = input_state
    ops_temp = final_state
    d_matrix = np.eye(len(input_state))
    t_list = []
    while(not np.array_equal(ips_temp, ops_temp) and count <= 1200):

        k_val, t_val, = k_t_transform(ops_temp, ips_temp)

        d_matrix_temp = create_t_matrix(t_val, 0, k_val, ips_temp)
        ops_temp = np.matmul(d_matrix_temp, ops_temp)
        #print(ops_temp)
        small_identity_matrix = np.eye(count)
        s_temp = len(ops_temp)

        t_matrix = np.block([[small_identity_matrix, np.zeros((count, s_temp))],
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


def comm_round_op_state(final_state, ini_state):
    """
    function for finding the target states for each round of communication
    in the order going from initial state to the final state

    returns: ops_compact_list which is ouput states without zeros
    """

    ds_func_output = create_ds_tlist(final_state, ini_state)
    ts_list = ds_func_output[1]
    ops_temp = final_state

    ops_list = []
    ops_list.append(ops_temp)
    for i, ele in enumerate(ts_list):

        ops_temp = np.matmul(ele, ops_temp)
        #ops_temp = np.sort(ops_temp)[::-1]
        ops_list.append(ops_temp)
        ini_state = ops_temp

    ops_list.reverse()
    ops_compact_list = []
    for i, ele in enumerate((ops_list)):
        ops_compact_list.append(ele[ele != 0])

    return ops_compact_list


def round_povm_func(round_ops_list):
    """
    povm function for all the rounds

    returns: list of povm and ooutput state density matrix
    """

    povm_round_list = []
    ops_dm_list = []
    for i in range(len(round_ops_list)-1):

        ip_state = round_ops_list[i]
        op_state = round_ops_list[i+1]
        op_state = concat_zeros(op_state, ip_state)
        output_locc_povm_func = locc_povm_func(op_state, ip_state)
        povm_out_list = output_locc_povm_func[0]
        perm_out_list = output_locc_povm_func[2]

        output_locc_povm_on_state = locc_povm_on_state(ip_state,  povm_out_list, perm_out_list)
        ops_obtained_list = output_locc_povm_on_state[0]
        prob_obtained_list = output_locc_povm_on_state[1]

        povm_round_list.append(povm_out_list)

        ops_dm = 0

        assert len(ops_obtained_list) != 0

        #if len(ops_obtained_list) == 0:
        #   raise Exception ("output state failed to obtained")

        num_ops_obtained = len(ops_obtained_list)
        for i in range(num_ops_obtained):
            ops_dm += prob_obtained_list[i]*vec2dm(ops_obtained_list[i])
        ops_dm_list.append(ops_dm)

    return povm_round_list, ops_dm_list

def one_round_povm_func(op_state, ip_state):
    """
    povm function for just one round

    returns: list of povm and ooutput state density matrix
    """

    op_state = concat_zeros(op_state, ip_state)
    output_locc_povm_func = locc_povm_func(op_state, ip_state)
    povm_out_list = output_locc_povm_func[0]
    perm_out_list = output_locc_povm_func[2]

    output_locc_povm_on_state = locc_povm_on_state(ip_state,  povm_out_list, perm_out_list)
    ops_obtained_list = output_locc_povm_on_state[0]
    prob_obtained_list = output_locc_povm_on_state[1]

    #povm_out_list, prob_out_list, perm_out_list = locc_povm_func(op_state, ip_state)
    #ops_obtained_list, prob_obtained_list
    #= locc_povm_on_state(ip_state,  povm_out_list, perm_out_list)

    #povm_round_list.append(povm_out_list)

    ops_dm = 0
    if len(ops_obtained_list) == 0:
        raise Exception ("output state failed to obtained")

    for i, ele in enumerate(ops_obtained_list):
        ops_dm += prob_obtained_list[i]*vec2dm(ele)
    #ops_dm_list.append(ops_dm)

    return povm_out_list, perm_out_list, output_locc_povm_func[1], ops_dm

def unitary_on_aux(povm_set):
    """
    takes the set of dxd dimensional povm with only two elements
    and makes a unitary to be applied on the auxiliary qubit

    returns: a list of the unitaries equal to the dimensions of the povms
    """

    assert len(povm_set) == 2

    dim_povm_ele = np.shape(povms[0])[0]


    unitary_final = 0
    unitary_list = []
    for i in range(dim_povm_ele):

        povm_ele0 = povms[0]
        povm_ele1 = povms[1]

        unitary_temp = np.zeros((2,2))

        unitary_temp[0,0] = np.sqrt(povm_ele0[i,i])
        unitary_temp[1,0] = np.sqrt(povm_ele1[i,i])
        unitary_temp[0,1] = 1*np.sqrt(povm_ele1[i,i])
        unitary_temp[1,1] = -1*np.sqrt(povm_ele0[i,i])

        unitary_temp = Qobj(unitary_temp)
        unitary_list.append(unitary_temp)
    return unitary_list

def total_unitary(num_qubits, unitary_list):
    """
    takes in the number of data qubits and list of the unitaries to make
    a unitary on the space of all the data qubits and auxiliary qubits

    returns: unitary on all the qubits
    """

    unitary_final = 0
    num_unitaries = len(unitary_list)
    for i in range(num_unitaries):

        binary_val = np.binary_repr(i, width=num_qubits)
        qubit = ket2dm(basis(2, int(binary_val[0])))
        for j in range(num_qubits-1):
            initialized_qubit1 = basis(2, int(binary_val[j+1]))
            qubit = tensor(qubit, ket2dm(initialized_qubit1))

        unitary_temp = tensor(qubit, unitary_list[i])
        unitary_final += unitary_temp

    return unitary_final

if __name__ == "__main__":

    ips = [0.7, 0.3]
    ips = self_tensor_prod(ips, 3)

    ops = [0.5, 0.5]
    ops = concat_zeros(ops, ips)


    ops_list1 = comm_round_op_state(ops, ips)
    #povm_round_list, ops_dm_list = round_povm_func(ops_list)
    #print(len(ops_list))
    number_of_communication_rounds = len(ops_list1)-1

    for k in range(number_of_communication_rounds):#len(ops_list)-1
        #print(j)
        inp_state = ops_list1[k]
        oup_state = ops_list1[k+1]

        """for each round we use the one_round_povm_func to get the povms
        of that round"""

        output = one_round_povm_func(oup_state, inp_state)
        povms = output[0]
        permutations = output[1]
        #print(povms[0])
        #print(povms[1])

        """for each comm round we get a list of unitaries
        these unitaries are orthogonal and are used to make the final unitary"""

        ut_list = unitary_on_aux(povms)

        """we get the final unitary for initializing data+auxiliary qubits"""
        ut = total_unitary(3, ut_list)
        print(len(ut_list))
        print(ut.dag()*ut)
        #print(len(povms))

        """measuring the aux qubits"""



        """

        number_qubits = int(np.ceil(np.log2(len(ips))))
        print(number_qubits)
        binary_val = np.binary_repr(i, width=number_qubits)
        #print(binary_val)
        #binary_val = f'{i:03b}'
        #print(i)
        #print(binary_val[0])
        #print(binary_val[1])

        qubit = 1

        for ii in range(number_qubits-1):

            initialized_qubit0 = basis(2, int(binary_val[ii]))
            initialized_qubit1 = basis(2, int(binary_val[ii+1]))
            qubit = tensor(ket2dm(initialized_qubit0), ket2dm(initialized_qubit1))


        #qubit0_dm = ket2dm(basis(2, int(binary_val[0])))
        #qubit1_dm = ket2dm(basis(2, int(binary_val[1])))
        #qubit2_dm = ket2dm(basis(2, int(binary_val[2])))
        #print(state_qubit0)

        unitary_final += tensor(qubit, unitary_temp)

        #print(unitary_final*unitary_final.dag())
    #print(ops_list)
    #print(ops_dm_list)

    #print()
    #print("fafa")

    #for i, ele in enumerate(povm_round_list):
    #
    #    print(ele[0])
    #    print(ele[1])





    #print(len(povm_round_list))

    #print(povm_round_list[0][1]+povm_round_list[0][0])
    #print(povm_round_list[0][0])
    #print(povm_round_list[1][1]+povm_round_list[1][0])
    #print(povm_round_list[1][0])
    #print(povm_round_list[2][1]+povm_round_list[2][0])
    #print(povm_round_list[2][0])
        #print(gamma_density_mat)
"""
