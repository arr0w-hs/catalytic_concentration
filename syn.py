#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 09:34:54 2024

@author: hsharma4
"""

import sys
import os
import time

import pickle
import matplotlib.pyplot as plt
import numpy as np
from qutip import *
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

#from bqskit.passes import LEAPSynthesisPass as sp
#from base_locc import *
from base_slocc import self_tensor_prod, concat_zeros, func_for_gamma
#from base_state_change import *
from base_siv_state_prep import err_cenotn
#from base_catalyst import *

from base_locc_alt import locc_operations

from qiskit import QuantumCircuit
from qiskit.compiler import transpile

from bqskit import compile
from bqskit.ir.circuit import Circuit
from bqskit.ir import Gate, Operation
from bqskit.ir.gates import CNOTGate, CRXGate, CZGate

from bqskit.compiler import Compiler
from bqskit.passes import ForEachBlockPass
from bqskit.passes import QFASTDecompositionPass
from bqskit.passes import ScanningGateRemovalPass
from bqskit.passes import UnfoldPass
from bqskit.qis import UnitaryMatrix


def err_sq_gate(input_gate, error_rate):
    """
    function for adding under_over_rotations to single qubnit gates
    """
    i = 0
    #print(input_gate.get_unitary(), "ig")
    #print(type(input_gate))
    gate_params = input_gate.params
    for ele_param in gate_params:
        gate_params[i] += np.random.normal(loc = 0., scale = error_rate)
        i += 1
        
    input_gate.params = gate_params
    #input_gate_unitary = 0
    #print(input_gate.get_unitary(),"og")
    #print(type(input_gate))
    return input_gate

def err_cnot_circuit(error_rate):
    """
    function for adding under_over_rotations to single qubnit gates
    """
    err_cnot = err_cenotn(error_rate)
    err_cnot_cir = compile(err_cnot.full(), max_synthesis_size = int(2))
    
    return err_cnot_cir


def sq_gate_error(gate_params, error_rate):
    """
    function for adding under_over_rotations to single qubnit gates
    """
    
    i = 0
    for ele_param in gate_params:
        gate_params[i] += np.random.normal(loc = 0., scale = error_rate)
        i += 1
  
    return gate_params


def error_circuit(input_circuit, sq_err, cnot_err):
    output_circuit = Circuit(input_circuit.num_qudits, input_circuit.radixes)
    
    for ele in input_circuit:
        if str(ele.gate) == "CNOTGate":
            #print(type(ele))
            print(input_circuit.point(ele))
            replacement_point = input_circuit.point(ele)
            replacement_gate = Gate(err_cenotn(cnot_err).full())
            replacement_operation = Operation(replacement_gate, ele.location)
            print(type(replacement_operation), "op_type")
            input_circuit.replace(replacement_point, replacement_operation)
            
            #err_cnot_cir = err_cnot_circuit(cnot_err)
            #output_circuit.append_circuit(err_cnot_cir, ele.location)
        elif str(ele.gate) == "U3Gate":
            #print(type(ele))
            err_sqg = err_sq_gate(ele, sq_err)
            

            #output_circuit.append_gate(err_sqg, ele.location)
    
    return output_circuit
    


def error_unitary(unitary, sq_error_rate, cnot_error_rate, catalyst_flag):
    """
    inputs:
    unitary is the input unitary (not Qobj)
    error_rate: error in degrees
    catalyst_flag: tells if there is catalyst or not, catalyst_flag == 1 means there is a catalyst
    
    returns: unitary with under and over rotations which is not a Qobj
    """
    assert catalyst_flag == 0 or catalyst_flag == 1
    
    syn_circuit = compile(unitary, max_synthesis_size = int(3+catalyst_flag))
    syn_circuit.save('syn_circuit_4povm.qasm')
    print(syn_circuit.gate_counts)
    err_circuit = error_circuit(syn_circuit, sq_error_rate, cnot_error_rate)
    print(err_circuit.gate_counts)
    
    output_unitary = err_circuit.get_unitary()
    
    return output_unitary


if __name__ == "__main__":
    #err_cnot_circuit(0.01)
    x = []
    y = []
    start_time = time.time()
    #main()
    #print(CRXGate.qasm_name)
    #print(i)
    #xx = 0.5+i/1000
    #x.append(xx)
    xx = 0.15
    ops = [0.5, 0.5]
    ips = [xx, 1-xx]
    
    
    ips = self_tensor_prod(ips, 2)
    ops = concat_zeros(ops, ips)
    
    gamma_ideal = func_for_gamma(ops, ips)
    #print(gamma_ideal)
    #povm_out_list, prob_out_list_junk, perm_out_list = locc_povm_func(gamma_ideal,
    #                                                         ips)
    operations = locc_operations(gamma_ideal, ips)
    print(len(operations))
    
    
    ini_uni = operations[0][0].full()
    #print((ini_uni))
    #ini_uni = err_cenotn(0.1).full()
    
    uni = np.asarray(ini_uni)
    #print(uni)
    ele_qobj = Qobj(uni)
    #ele_qobj = ele_qobj.tidyup()
    
    o_uni = error_unitary(ini_uni, 0, 0, 0)
    print(o_uni)
    oput = Qobj(np.asarray(o_uni))#.tidyup()
    xxxx = (ele_qobj*oput.dag())
    print(xxxx.tidyup())
        #print(ele_qobj)
    """
    er = CRXGate()
    #er.params = [0]
    print(er.get_unitary([np.pi]))
    """
    
    #y.append(len(operations))
    #for i, ele in enumerate((povm_out_list)):
    #    vertices = np.nonzero(ele)
    #    print(vertices)
    #plt.plot(x,y)
    #print(len(povm_out_list), "number of operators in povm")
    #print(povm_out_list[0])
    
    syn_circuit1 = Circuit(3)
    syn_circuit = Circuit(3)
    
    #print(np.shape(syn_circuit.get_unitary()))
    #print(syn_circuit)
    #syn_circuit.compress()
    
    #syn_circuit = compile(ut, max_synthesis_size = 3)
    #gate_dict = syn_circuit.gate_counts
    #print(gate_dict, "gate_dict")
    #cnot_count = list(gate_dict.values())[1]
    #print(cnot_count)
    #print(syn_circuit.coupling_graph)
    #syn_circuit.save('syn_circuit_4povm.qasm')
    #print(syn_circuit)
    #ini_uni = syn_circuit.get_unitary()
    #print(ini_uni)
    
    
    #print(o_uni)
    
    #uni = ele.get_unitary()
    
    #syn_circuit1.from_unitary(ut)
    #gate_dict = syn_circuit1.gate_counts
    #cnot_count = list(gate_dict)
    #print(gate_dict)
    #print(syn_circuit.gate_counts)
    #print(syn_circuit.num_cycles)
    #print(syn_circuit.depth)
    
    
    bqs_time = time.time()
    
    """
    gate_count = []
    depth = []
    x = []
    print(len(perm_out_list))
    for i, elem in enumerate(perm_out_list):
        print(i)
        x.append(i)
        syn_circuit = compile(elem)
        gate_count.append(syn_circuit.gate_counts)
        depth.append(syn_circuit.depth)
      """  
        
    """
    # Let's create a random 3-qubit unitary to synthesize and add it to a
    # circuit.
    circuit = Circuit.from_unitary(perm_out_list[0])
    
    # We now define our synthesis workflow utilizing the QFAST algorithm.
    workflow = [
        QFASTDecompositionPass(),
        ForEachBlockPass([
            LEAPSynthesisPass(),  # LEAP performs native gate instantiation
            ScanningGateRemovalPass(),  # Gate removal optimizing gate counts
        ]),
        UnfoldPass(),
    ]
    
    # Finally let's create create the compiler and execute the CompilationTask.
    with Compiler() as compiler:
        compiled_circuit = compiler.compile(circuit, workflow)
        print(compiled_circuit.gate_counts)
        print(compiled_circuit.depth)
    
    
    #sp.synthesize(self, utry =I, data =0)
    
    
    num_qubits = 3
    qc = QuantumCircuit(num_qubits)
    qc.unitary(ut, qubits=range(num_qubits))
    qc.draw('mpl')
    qc = transpile(qc, optimization_level=3, basis_gates=["u3", "cx"])
    print(dict(qc.count_ops()))
    """
    """for ele in syn_circuit:
        #print(ele)
        if str(ele.gate) != "CNOTGate":
            param = ele.params
            epsilon = np.pi/180*error_rate
            param = sq_gate_error(param, epsilon)
            ele.params = param
            
            #print(ele)
        else:
            
            #print(ele.get_qasm())
            ele._gate = CZGate()#CRXGate()
            print(ele.params)
            #ele.params = [0.1]
            #print(ele.dim)
            print(ele.gate)
            #print(ele.get_unitary())
            print(ele.location, "loc")
            
            #ele.gate = CRXGate()
            
            #ele.replace(point, op)
    """
    
    #print("--- %s seconds ---" % (bqs_time - start_time))
    print("--- %s seconds ---" % (time.time() - start_time))
