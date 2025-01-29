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
import qutip as qt
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)
from base_slocc import concat_zeros, self_tensor_prod
from base_locc import locc_povm_func, k_t_transform, create_t_matrix, locc_povm_on_state
#from base_state_change import vec2dm

sys.path.append(os.path.dirname(__file__))

def vec2dm(vector):
    vector1 = np.zeros(len(vector))
    for i in range(len(vector)):
       vector1[i] = np.sqrt(vector[i])
    dm = np.outer(vector1, np.atleast_2d(np.conjugate(vector1)).T)
    #print(dm)
    return dm

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
    #print(final_state)
    #print(input_state)
    count = 0
    assert (len(final_state) <=  len(input_state))
        #raise Exception("Incoherent dimensions of states")

    ips_temp = input_state
    ops_temp = final_state
    d_matrix = np.eye(len(input_state))
    t_list = []
    while(not np.array_equal(ips_temp, ops_temp) and count <= 1200):

        k_val, t_val, = k_t_transform(ops_temp, ips_temp)

        #print(k_val, t_val)
        #if t_val == 1:
        #   continue

        d_matrix_temp = create_t_matrix(t_val, 0, k_val, ips_temp)
        ops_temp = np.matmul(d_matrix_temp, ops_temp)
        #print(ops_temp)
        small_identity_matrix = np.eye(count)
        s_temp = len(ops_temp)

        t_matrix = np.block([[small_identity_matrix, np.zeros((count, s_temp))],
                  [np.zeros((s_temp, count)), d_matrix_temp]
                  ])
        #print(t_matrix)
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
    ops_list.append(np.asarray(ops_temp))
    for ele in (ts_list):
        #print(ele)
        ops_temp = np.matmul(ele, ops_temp)
        #ops_temp = np.sort(ops_temp)[::-1]
        ops_list.append(ops_temp)
        ini_state = ops_temp
    #print(ops_list)
    ops_list.reverse()
    ops_compact_list = []
    for i, ele in enumerate((ops_list)):
        ops_compact_list.append(ele[ele != 0])

    #print(" ")
    #print(ops_compact_list)
    return ops_compact_list


def one_round_povm_func(op_state, ip_state):#, num_qubits):
    """
    povm function for just one round

    returns: list of povm and output state density matrix
    """
    #print(len(ip_state))
    #dummy_inital_state = [1]*2**num_qubits
    #ip_state = concat_zeros(ip_state, dummy_inital_state)
    op_state = concat_zeros(op_state, ip_state)
    output_locc_povm_func = locc_povm_func(op_state, ip_state)
    povm_out_list = output_locc_povm_func[0]
    perm_out_list = output_locc_povm_func[2]
    #print(len(perm_out_list))
    for i, ele in enumerate(perm_out_list):
        #print(type(ele))
        perm_out_list[i] = np.transpose(ele)
    #print(perm_out_list)





    output_locc_povm_on_state = locc_povm_on_state(ip_state,  povm_out_list, perm_out_list)
    ops_obtained_list = output_locc_povm_on_state[0]
    prob_obtained_list = output_locc_povm_on_state[1]

    #povm_out_list, prob_out_list, perm_out_list = locc_povm_func(op_state, ip_state)
    #ops_obtained_list, prob_obtained_list
    #= locc_povm_on_state(ip_state,  povm_out_list, perm_out_list)

    #povm_round_list.append(povm_out_list)

    ops_dm = 0

    assert len(ops_obtained_list) != 0
    #if len(ops_obtained_list) == 0:
    #    raise Exception ("output state failed to obtained")

    for i, ele in enumerate(ops_obtained_list):
        ops_dm += prob_obtained_list[i]*vec2dm(ele)
    #ops_dm_list.append(ops_dm)


    return povm_out_list, perm_out_list, output_locc_povm_func[1], ops_dm

def unitary_on_auxiliary(povm_set):
    """
    takes the set of dxd dimensional povm with only two elements
    and makes a list of unitary to be applied

    returns: a list of the unitaries equal to the dimensions of the povms
    """
    #print((povm_set))
    assert len(povm_set) == 2

    dim_povm_ele = np.shape(povm_set[0])[0]

    unitary_list = []
    for i in range(dim_povm_ele):

        povm_ele0 = povm_set[0]
        povm_ele1 = povm_set[1]

        unitary_temp = np.zeros((2,2))

        unitary_temp[0,0] = np.sqrt(povm_ele0[i,i])#a
        unitary_temp[1,0] = np.sqrt(povm_ele1[i,i])#c
        unitary_temp[0,1] = 1*np.sqrt(povm_ele1[i,i])#b
        unitary_temp[1,1] = -1*np.sqrt(povm_ele0[i,i])#d
        unitary_temp = np.real(unitary_temp)
        # print(unitary_temp)
        unitary_temp = qt.Qobj(unitary_temp)
        unitary_list.append(unitary_temp)

    return unitary_list

def one_round_unitary_list(num_qubits, unitary_list):
    """
    takes in the number of data qubits and list of the unitaries to make
    a unitary on the space of all the data qubits and auxiliary qubits

    returns: list of controlled-unitaries that multiply to each other to 
    give on big unitary on all the data qubits + auxiliary
    """


    max_num_unitaries = int(2**num_qubits)
    num_unitaries = len(unitary_list)
    # print(num_unitaries, "num_unitaries")
    uni_list = []

    for k in range(num_unitaries):
        unitary_final = 0
        for i in range(max_num_unitaries):
            binary_val = np.binary_repr(i, width=num_qubits)
            # print(binary_val)
            """qubit dm is the state of the data qubits for a certain unitary"""
            qubit_dm = qt.ket2dm(qt.basis(2, int(binary_val[0])))
            for j in range(num_qubits-1):
                initialized_qubit1 = qt.basis(2, int(binary_val[j+1]))
                qubit_dm = qt.tensor(qubit_dm, qt.ket2dm(initialized_qubit1))
    
            """unitary temp is the tensor product of data qubit initializations
               and the unitary on the data qubit"""
            if i == k:
                unitary_temp = qt.tensor(qubit_dm, unitary_list[i])
            else:
                unitary_temp = qt.tensor(qubit_dm, qt.qeye(2))
                # continue
            """unitary final is the unitary on data + aux qubits"""
            unitary_final += unitary_temp
        I = qt.qeye(2)
        uni_list.append(unitary_final)
        # unitary_final = qt.tensor(I,I,I,I)
        # for ele in uni_list:
        #     unitary_final *= ele


    # print(len(uni_list))
    # print(uni_list)

    return uni_list

def one_round_unitary(num_qubits, unitary_list):
    """
    takes in the number of data qubits and list of the unitaries to make
    a unitary on the space of all the data qubits and auxiliary qubits

    returns: big unitary on all the data qubits + auxiliary
    """

    unitary_final = 0
    max_num_unitaries = int(2**num_qubits)
    num_unitaries = len(unitary_list)
    # print(num_unitaries, "num_unitaries")

    for i in range(max_num_unitaries):
        binary_val = np.binary_repr(i, width=num_qubits)
        # print(binary_val)
        """qubit dm is the state of the data qubits for a certain unitary"""
        qubit_dm = qt.ket2dm(qt.basis(2, int(binary_val[0])))
        for j in range(num_qubits-1):
            initialized_qubit1 = qt.basis(2, int(binary_val[j+1]))
            qubit_dm = qt.tensor(qubit_dm, qt.ket2dm(initialized_qubit1))

        """unitary temp is the tensor product of data qubit initializations
           and the unitary on the data qubit"""
        if i < num_unitaries:
            unitary_temp = qt.tensor(qubit_dm, unitary_list[i])
        else:
            unitary_temp = qt.tensor(qubit_dm, qt.qeye(2))
            # continue
        """unitary final is the unitary on data + aux qubits"""
        unitary_final += unitary_temp
        # uni = np.real((unitary_final*unitary_final.dag()).tidyup().full())
        # # uni = [int]
        # print(uni)

    return unitary_final

def locc_operations(final_state, input_state):
    """
    function which takes in the output and input states

    returns: for each of the rounds it returns
            (1) unitary qobj on data qubits and auxiliary qubits
            (2) two permutations corresponding to two measurement results of auxiliary
    """
    num_qubits = int(np.ceil(np.log2(len(input_state))))
    out_state_list = comm_round_op_state(final_state, input_state)
    #print(out_state_list)
    number_of_communication_rounds = len(out_state_list)-1

    operation_list = []

    for k in range(number_of_communication_rounds):
        one_round_operation = []
        inp_state = out_state_list[k]
        #print(inp_state, "in state")
        out_state = out_state_list[k+1]
        #print(out_state, "out state")
        #print(" ")
        """for each round we use the one_round_povm_func to get the povms
        of that round"""
        #print(out_state)
        #print(inp_state)
        output = one_round_povm_func(out_state, inp_state)#, num_qubits)
        povms = output[0]
        permutations = output[1]
        # for ele in povms:
        #     print(np.shape(ele), "ele")

        #print(permutations)

        """we continue in case povms have only one element that is identity"""
        if len(povms) == 1:
            #print(np.shape(povms[0]))
            print(np.array_equal(povms[0], qt.qeye(np.shape(povms[0])[0]))
                  ,  "the povm element is identity")
            #print(povms[0])
           # print(permutations)
            #one_round_operation.append(povms)
            #one_round_operation.append(permutations)
            #print("identity in povm elements")
            #print(permutations)
            continue
        """for each comm round we get a list of unitaries
        these unitaries are orthogonal and are used to make
        the final unitary list"""

        ut_list = unitary_on_auxiliary(povms)

        """we get the final unitary for initializing data+auxiliary qubits
        for each round of communication separately"""
        ut_final_list = one_round_unitary_list(num_qubits, ut_list)
        # ut_final1 = one_round_unitary(num_qubits, ut_list)
        # print((ut_final*ut_final1.dag()).tidyup())

        #print(ut.dag()*ut)
        #print(len(povms))
        one_round_operation.append(ut_final_list)
        one_round_operation.append(permutations)
        operation_list.append(one_round_operation)
        """measuring the aux qubits"""
        # print()

    #print((operation_list))
    #assert len(operation_list) == number_of_communication_rounds
    return operation_list


def slocc_unitary(slocc_povm):
    """function for giving out the slocc unitary"""
    num_qubits = int(np.ceil(np.log2(np.shape(slocc_povm[0])[0])))
    #print(len(slocc_povm))
    ut_list = unitary_on_auxiliary(slocc_povm)

    """we get the final unitary for initializing data+auxiliary qubits
    for each round of communication separately"""
    slocc_uni = one_round_unitary(num_qubits, ut_list)

    #print(slocc_unitary*slocc_unitary.dag())

    return slocc_uni


if __name__ == "__main__":

    ips = [0.85, 0.15]
    ips = self_tensor_prod(ips, 3)

    ops = [0.6141249999999999, 0.385875]
    ops = concat_zeros(ops, ips)

    locc_operations(ops, ips)


