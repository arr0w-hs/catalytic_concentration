#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 28 17:40:41 2025

@author: hsharma4
for compilation of locc unitary
the locc operations in this list are 
in a list whose product gives the final unitary
"""



import sys
import os
from pathlib import Path


import numpy as np
import pandas as pd
import qutip as qt
import argparse

from qutip.qip.operations import expand_operator
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

from base_transform import  pre_conversion_process, schmidt_decomp_of_dm
from base_slocc import concat_zeros, func_for_gamma, slocc_povm_func
#from base_siv_state_prep import basis2schmidt
from qutip.qip.operations import cnot

from base_locc_alt import locc_operations, slocc_operations

from base_depol_channels import new_state_pauli_x1#new_state_depol, new_state_pauli_z,

from bqskit.compiler import Compiler#, GateSet, CompilationTask
from bqskit.ir.circuit import Circuit
#from bqskit.passes import ForEachBlockPass, LEAPSynthesisPass
from bqskit.passes import QSearchSynthesisPass#, QFASTDecompositionPass, ScanningGateRemovalPass, UnfoldPass

from bqskit.passes import SimpleLayerGenerator#, WideLayerGenerator
#from bqskit.ir.gates import ISwapGate, PauliGate, CCPGate, RC3XGate
from bqskit.ir.gates import U3Gate, CNOTGate#,RXXGate, RYYGate, RZZGate
#from bqskit.passes.search import LayerGenerator

parser = argparse.ArgumentParser(description="round number")
parser.add_argument("--i", type=int, help="index of round list")
args, _ = parser.parse_known_args()
round_number = args.i

zero = qt.basis(2,0)
one = qt.basis(2,1)
I = qt.qeye(2)
X = qt.sigmax()
Z = qt.sigmaz()
Y = qt.sigmay()
H = 1/np.sqrt(2)*(X+Z)

ts = pd.Timestamp.today(tz = 'Europe/Stockholm')
date_str = str(ts.date())
time_str = ts.time()
time_str = str(time_str.hour)+ str(time_str.minute) + str(time_str.second)
# print(time_str)

# #data_directory = os.path.join(dir_name+"/test_circuits", date_str+"/")
# data_directory = os.path.join(dir_name+"/test_circuits", date_str+"_cat_disti_comparison_nc/")
# #plots_directory = os.path.join(dir_name+"/plots", date_str+"_cat_disti_comparison/")

# date_folder = Path(data_directory)
# if date_folder.exists():
#     print("date folder exists")
# else:
#     os.mkdir(data_directory)

round_number = 5

def basis2schmidt(psn_st):
    psn_dims = psn_st.dims
    psn_data = psn_st.full()
    """removing the photon state, as it is already in |0>"""
    sn_data = np.array_split(psn_data, 2)[0]
    psn_length = sn_data.shape[0]
    psn_length = int(np.sqrt(psn_length))
    sn_matrix_form = np.reshape(sn_data, (psn_length, psn_length))

    U, S, Vh = np.linalg.svd(sn_matrix_form, full_matrices=True)
    #print(np.allclose(sn_matrix_form, np.dot(U* S, Vh)))
    s, u, v = qt.Qobj(S), qt.Qobj(U).dag(), qt.Qobj(Vh).dag()
    basis_matrix = qt.tensor( I, u, v.trans()).full()
    basis_matrix = qt.Qobj(basis_matrix, dims = [psn_dims[0], psn_dims[0]])

    psn = psn_st.full()
    psn = qt.Qobj(psn, dims = psn_dims)#
    psn_schmidt_basis = (basis_matrix*psn).tidyup()
    #psn_schmidt_basis = psn_schmidt_basis.full()
    #psn_schmidt_basis = qt.Qobj(psn_schmidt_basis, dims = psn_dims)

    return psn_schmidt_basis, basis_matrix, s


def compile_unitary(in_unitary):
    """compiling the unitary into a circut"""
    #print(in_unitary*in_unitary.dag())
    # in_unitary = in_unitary.full()
    # print(np.shape(in_unitary))
    # t1=  time.time()
    # compiled_circuit = compile(in_unitary, max_synthesis_size = int(4))
    # print("compiled", time.time()-t1, "seconds")


    circuit = Circuit.from_unitary(in_unitary)#.full())
    
    # # We now define our synthesis workflow utilizing the QFAST algorithm.
    # workflow = [
    #     QFASTDecompositionPass(),
    #     ForEachBlockPass([
    #         LEAPSynthesisPass(),  # LEAP performs native gate instantiation
    #         ScanningGateRemovalPass(),  # Gate removal optimizing gate counts
    #     ]),
    #     UnfoldPass(),
    # ]
    
    # # Finally let's create create the compiler and execute the CompilationTask.
    # with Compiler() as compiler:
    #     compiled_circuit = compiler.compile(circuit, workflow)
    #     print(compiled_circuit.gate_counts)
    
    # len_cirq = (len(compiled_circuit))
    #syn_circuit.compress()
    # layer_gen = WideLayerGenerator(multi_qudit_gates=CCXGate(), single_qudit_gate=U3Gate(1))
    layer_gen = SimpleLayerGenerator(two_qudit_gate=CNOTGate(), single_qudit_gate_1=U3Gate())

    configured_qsearch_pass = QSearchSynthesisPass(layer_generator=layer_gen)
    
    # Create and execute a compilation task
    with Compiler() as compiler:
        # task = CompilationTask()
        compiled_circuit = compiler.compile(circuit, [configured_qsearch_pass])
    
    for gate in compiled_circuit.gate_set:
        print(f"{gate} Count:", compiled_circuit.count(gate))

    # print(GateSet.multi_qudit_gates())
    print(compiled_circuit.gate_counts)

    return compiled_circuit

def extend_perm(perm_list, num_qubits):
    """function for extending the dimensions of permutation unitaries
       to dimensions of the initial state

       inputs:
           1. list of permutations
           2. initial state density matrix with only alices qubits

       returns: list of extended permutation unitaries
    """

    #initial_dims = initial_state_dm.dims
    #initial_state_dm = initial_state_dm.full()

    #print(initial_dims)
    #print(initial_state)
    required_len = int(2**num_qubits)
    initial_dims = [[2]*num_qubits, [2]*num_qubits]

    #print(required_len)
    extended_perm_list = []
    for i in range(len(perm_list)):

        perm_mat = perm_list[i]
        perm_len = np.shape(perm_mat)[0]
        # print(perm_mat)
        extra_len = int(required_len-perm_len)

        identity_temp = np.eye(extra_len)

        side_zeros = np.zeros((perm_len, extra_len))
        lower_zeros = np.zeros((extra_len, perm_len))
        #print(identity_temp, side_zeros, lower_zeros)
        extended_perm_mat = np.block([[perm_mat, side_zeros],
                  [lower_zeros, identity_temp]
                  ])

        extended_perm_mat = qt.Qobj(extended_perm_mat, dims = initial_dims)
        extended_perm_list.append(extended_perm_mat)
        print(extended_perm_mat)

    return extended_perm_list



def unitary2circ(list_unitaries, cat_flag):
    """list has 4 lements:
        1) list of operations for LOCC
        2) SLOCC matrix
        3) u matrix for alice for schmidt conversion
        4) v matrix for bob for schmidt conversion
    """
    operations = list_unitaries[0]
    operations_out = []
    num_rounds = len(operations)
    metadata_list = []

    u_circ = compile_unitary(list_unitaries[2])
    v_circ = compile_unitary(list_unitaries[3])
    # print(u_circ)
    # print(v_circ)
    # print(u_circ.gate_counts)
    # print(v_circ.gate_counts)

    slocc_uni = slocc_operations(list_unitaries[1])
    # print(np.real(slocc_uni.full()))
    slocc_circ = 0#compile_unitary(qt.Qobj(slocc_uni))

    for i, ele in enumerate(operations):
        # continue
        # if i != round_number:
            # continue


        povm_circ_list = []
        one_round_circ = []
        povm_unitary_list = ele[0]
        perm_list = extend_perm(ele[1], 2+cat_flag)

        for ii, eleme in enumerate(povm_unitary_list):
            # metadata_list.append( (i, ii))
            # with open(data_directory + time_str +'_metadata.txt', mode="w") as f:
            #     f.write(str(metadata_list))
            #     f.close()
            print(i,ii)
            povm_circ_list.append(compile_unitary(eleme))


        for j, elem in enumerate(perm_list):
            perm_list[j] = compile_unitary(qt.Qobj(elem))


        # time1 = time.time()


        one_round_circ.append(povm_circ_list)
        one_round_circ.append(perm_list)

        operations_out.append(one_round_circ)

        #else:
        #    continue

    return operations_out, slocc_circ, u_circ, v_circ




def depol_channel(rho_in, err_prob, qubit_loc, num_qubits):
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

    gates = [X, Z]
    for i, ele in enumerate(gates):
        #gates[i] = gate_expand_1toN(ele, num_qubits, qubit_loc)
        #print(qubit_loc)
        gates[i] = expand_operator(ele, dims=[2]*num_qubits, targets=[qubit_loc])
        #print(gates[i])

    rho_out = (1-err_prob)*rho_in + err_prob/3*(gates[0]*rho_in*gates[0] +
                                                gates[0]*gates[1]*rho_in*gates[1]*gates[0] +
                                                gates[1]*rho_in*gates[1])
    return rho_out

def gate_unitary(element, num_qubits, bob_flag, cat_flag, perm_flag):
    """
    function for converting circuit element into a high dim qt.Qobj unitary

    inputs:
        element is the circuit element (not qt.Qobj)
        total num of qubits at alice's side (including ancilla)
        bob_flag == 1 means that the unitary acts on bob's qubits
    returns:
        output unitary in higher dim
    """
    #print(np.ceil(num_qubits/2), "asdfkh")
    bobs_qubits = 1 + bob_flag*(np.ceil(num_qubits/2)-1+perm_flag)
    #print(bobs_qubits, cat_flag, perm_flag)
    dimens = [2]*num_qubits


    if str(element.gate) == "CNOTGate":
        loc = list(element.location)

        #print(loc)
        #print([int(loc[0]+bobs_qubits), int(loc[1]+bobs_qubits)])
        #print(num_qubits)
        #there is +1 in loc due to the photon
        #output_uni = gate_expand_2toN(cnot(), num_qubits, control = int(loc[0]+1), target = int(loc[1]+1))
        #output_uni = gate_expand_2toN(cnot(), num_qubits, control = int(loc[0]+cat_flag+4), target = int(loc[1]+cat_flag+4))
        output_uni = expand_operator(cnot(), dims=dimens, targets=[int(loc[0]+bobs_qubits), int(loc[1]+bobs_qubits)])

    elif str(element.gate) == "U3Gate":
        loc = list(element.location)
        #print(loc)
        #print(int(loc[0]+bobs_qubits))
        #print(num_qubits)
        gate_uni = np.asarray(element.get_unitary())
        gate_uni = qt.Qobj(gate_uni)
        #output_uni = gate_expand_1toN(gate_uni, num_qubits, int(loc[0]+1))
        #output_uni = gate_expand_1toN(gate_uni, num_qubits, int(loc[0]+cat_flag+4))
        output_uni = expand_operator(gate_uni, dims=dimens, targets=[int(loc[0]+bobs_qubits)])
        #print(loc[0])
    else:
        print("other gate in the system")
        raise Exception

    return output_uni

def gate_depol(rho_in, element, sq_err, cnot_err, num_qubits, bob_flag, cat_flag, perm_flag):
    """
    function for acting with depol noise on the qubits after gate operation

    inputs:
        input dm
        element is the circuit element (not qt.Qobj)
        errors rates
        total num of qubits at alice's side (including ancilla)
        bob_flag == 1 means that the unitary acts on bob's qubits
    returns:
        output dm which is just depolarised
    """
    bobs_qubits = 1 + bob_flag*(np.ceil(num_qubits/2)-1+perm_flag)#+cat_flag
    dimens = [2]*num_qubits

    if str(element.gate) == "CNOTGate":
        loc = list(element.location)
        #rho_out = depol_channel(, cnot_err, int(loc[0]+1), num_qubits)
        #rho_out = depol_channel(rho_out, cnot_err, int(loc[1]+1), num_qubits)

        rho_out = depol_channel(rho_in, cnot_err, int(loc[0]+bobs_qubits), num_qubits)
        rho_out = depol_channel(rho_out, cnot_err, int(loc[1]+bobs_qubits), num_qubits)

    elif str(element.gate) == "U3Gate":
        loc = list(element.location)
        #rho_out = depol_channel(, sq_err, int(loc[0]+1), num_qubits)
        #print(loc[0]+cat_flag+bobs_qubits)
        #print(loc[0])
        rho_out = depol_channel(rho_in, sq_err, int(loc[0]+bobs_qubits), num_qubits)

    else:
        print("other gate in the system")
        raise Exception


    return rho_out



def depol_final_state(syn_circ, in_state, sq_error_rate, cnot_error_rate, num_qubits, bob_flag, cat_flag, perm_flag):
    """
    func for compiling the unitary and going through each gate and applying depol channel

    inputs:
        in_state is the dm of the input
        syn_circ is the input synthesised circuit (not qt.Qobj)
        error_rates: error  probability
        catalyst_flag: tells if there is catalyst or not,
                       catalyst_flag == 1 means there is a catalyst
       num_qubits: number of qubits the unitary acts on

    returns: gives the final output depolarised state
    """

    #assert catalyst_flag == 0 or catalyst_flag == 1
    """compiling the unitary into a circut"""
    #in_unitary = in_unitary.full()
    #syn_circuit = compile(in_unitary, max_synthesis_size = int(4))
    #syn_circuit.compress()

    #num_qubits = int(3+catalyst_flag)
    #output_unitary = gate_expand_1toN(qeye(2), num_qubits, 0)

    """applying depol channel after each gate"""


    out_state = in_state
    for ele in syn_circ:

        unitary = gate_unitary(ele, num_qubits, bob_flag, cat_flag, perm_flag)
        #print(unitary)
        #print(out_state.dims)
        out_state = unitary*out_state*unitary.dag()
        out_state = gate_depol(out_state, ele, sq_error_rate, cnot_error_rate, num_qubits, bob_flag, cat_flag, perm_flag)

    return out_state

def measure_aux(input_dm, num_qubits, cat_flag):
    """aux is at the (4+c)th location that is the (3+c)th index in the qobj

    returns: the output states after the two measurements"""
    dimens = [2]*num_qubits
    zero = qt.ket2dm(qt.basis(2,0))
    one_1 = X*qt.ket2dm(qt.basis(2,1)) #X is for taking the aux qubit to 0 after measurement

    #mea0 = gate_expand_1toN(zero, num_qubits, int(3+cat_flag))
    #mea1 = gate_expand_1toN(one, num_qubits, int(3+cat_flag))

    mea0 = expand_operator(zero, dims=dimens, targets=[int(3+cat_flag)])
    mea1 = expand_operator(one_1, dims=dimens, targets=[int(3+cat_flag)])
    #print(mea0)

    mea0_out = mea0*input_dm*mea0.dag()
    mea1_out = mea1*input_dm*mea1.dag()
    #mea0_out =

    return mea0_out, mea1_out


def to_be_syn_nc(prepared_dm):

    """pre_conv_process gives out the u and v matrices in catalytic conversion"""

    s_coeff, prepared_dm_schmidt_basis, pure_st, uv = schmidt_decomp_of_dm(prepared_dm)
    input_state_array = np.reshape(np.real(s_coeff), [4])

    output_state = [0.5, 0.5]
    output_state_array = concat_zeros(output_state, input_state_array)

    """use function for gamma to find the ideal gamma that is needed"""
    gamma_ideal = func_for_gamma(output_state_array, input_state_array)

    """use function for locc povms to find the ideal povms to get to gamma"""
    operations = locc_operations(gamma_ideal, input_state_array)


    """use function for slocc povms to find the ideal povms to get to final state"""
    slocc_povm = slocc_povm_func(output_state_array, input_state_array)

    return operations, slocc_povm, uv[0], uv[1]

def to_be_syn_cat(prepared_dm):

    """pre_conv_process gives out the u and v matrices in catalytic conversion"""
    output_state_qobj, psnc_dm, carbon_st, input_state_array, output_state_array, uv = pre_conversion_process(prepared_dm)

    """use function for gamma to find the ideal gamma that is needed"""
    gamma_ideal = func_for_gamma(output_state_array, input_state_array)
    #print(gamma_ideal)

    """use function for locc povms to find the ideal povms to get to gamma"""
    operations = locc_operations(gamma_ideal, input_state_array)

    """use function for slocc povms to find the ideal povms to get to final state"""
    slocc_povm = slocc_povm_func(output_state_array, input_state_array)

    # print(uv[0])
    # print(uv[1])


    return operations, slocc_povm, uv[0], uv[1]



if __name__ == "__main__":
    #err_cnot_circuit(0.01)
    #print(dudeness)


    #final_state, final_state_loss, prob_final_state, prob_final_loss, mea_value = prepare_dm_withreset(
    #        psn_dm, cnot_errore, rr, lvec, num_reset, sqe_error, dist)

    a = 0.9
    p = 0.95
    final_state = new_state_pauli_x1(a, p)
    # in_uni = np.zeros((4,4))
    # in_uni[0,1] =1
    # in_uni[1,3] =1
    # in_uni[2,0] =1
    # in_uni[3,2] =1
    # print(in_uni)

    # in_uni = qt.Qobj(in_uni, dims = [[2,2],[2,2]])
    # print(in_uni)
    # swap = qt.gates.swap()
    # A = (swap*qt.tensor(I,X))
    # print(A)
    # O = qt.ket2dm(one)
    # Z = qt.ket2dm(zero)
    # B = qt.tensor(Z, Z, X)+qt.tensor(O,O,X)
    # print(B)
    # C = qt.tensor(O,A)+qt.tensor(Z, A)
    # print(C)

    # a = np.zeros((8,8))
    # a[0,1] =1
    # a[1,0] =1
    # a[2,3] =1
    # a[3,5] =1
    # a[4,2] =1
    # a[5,4] =1
    # a[6,7] =1
    # a[7,6] =1
    # a = qt.Qobj(a, dims = [[2,2,2],[2,2,2]])
    # a = (a*qt.tensor(I,I,X))
    # print(a*qt.tensor(I,swap)*qt.tensor(I,X,I))
    # print(a)






    # cir = compile_unitary(a.full())
    # for ele in cir:
    #     print(ele)

    #print(final_state.shape, "final state shape")
    #catalytic_conversion(final_state)

    #_, psnc_dm, _, _, _, _ = pre_conversion_process(final_state)

    # list_unit = to_be_syn_nc(final_state)
    list_unit = to_be_syn_cat(final_state)
    #print(list_unit[1], list_unit[2], list_unit[3])
    # list_circs = unitary2circ(list_unit, 1)
    # print(list_circs)
    #povm_uni = list_unit[0][4][0]
    #print(povm_uni)

    #t1 = time.time()
    #c1 = compile_unitary((povm_uni))
    #print(time.time()-t1)
    #print(c1.gate_counts)
    #list_circs = split_circuit(povm_uni)

   # uni = list_unit[0][0][0]

    """
    print(uni.shape)
    t1 = time.time()
    circuit = Circuit.from_unitary(uni.full())

    model = MachineModel(circuit.num_qudits, gate_set={U3Gate(), CNOTGate()})

    # We now define our synthesis workflow utilizing the QFAST algorithm.
    workflow = [
        #QFASTDecompositionPass(),
        #QPredictDecompositionPass(),
        QSearchSynthesisPass(),
        ForEachBlockPass([
            LEAPSynthesisPass(),  # LEAP performs native gate instantiation
            #ScanningGateRemovalPass(),  # Gate removal optimizing gate counts
            ExhaustiveGateRemovalPass()
        ]),
        UnfoldPass(),
    ]

    # Finally let's create create the compiler and execute the CompilationTask.
    with Compiler() as compiler:
        circuit.MachineModel = model
        compiled_circuit = compiler.compile(circuit, workflow)
        print(compiled_circuit.gate_counts)
    print(time.time()-t1)

    #print(adf)
    #cir = compile_unitary(uni)
    #print(cir.gate_counts)


    #locc_oper = list_unit[0]
    #for ele in locc_oper:#
    #
    #    print(np.real(ele[0].full()))
    #    print()
    #print(list_unit)
    #list_circs = unitary2circ(list_unit, 0)
    #ps_aiu = add_aux(psnc_dm)
    #output_state = apply_locc_conversion(list_circs[0], ps_aiu, 1, 0, 0)
    #print(hj)

    """
    #print(list_circs)


    # data_dict = {
    #     "list_unitary": list_unit,
    #     "list_circs": list_circs,
    #     "a": a,
    #     "p": p,
    #     }

    # with open(data_directory+ time_str +'_' + str(round_number) + '.pkl', 'wb') as f:  # open a text file
    #     pickle.dump(data_dict, f)



