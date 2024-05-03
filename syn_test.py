#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:57:56 2024

@author: hsharma4
"""
import sys
import os

import pickle
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

from bqskit.passes import LEAPSynthesisPass as sp
from base_locc import *
from base_slocc import *
from base_state_change import *
from base_siv_state_prep import *
from base_catalyst import *

from qiskit import QuantumCircuit

from bqskit import compile

from bqskit.compiler import Compiler
from bqskit.ir.circuit import Circuit
from bqskit.passes import ForEachBlockPass
from bqskit.passes import LEAPSynthesisPass
from bqskit.passes import QFASTDecompositionPass
from bqskit.passes import ScanningGateRemovalPass
from bqskit.passes import UnfoldPass
from bqskit.qis import UnitaryMatrix




ops = [0.5, 0.5]
ips = [0.85, 0.15]


ips = self_tensor_prod(ips, 3)
ops = concat_zeros(ops, ips)

gamma_ideal = func_for_gamma(ops, ips)

povm_out_list, prob_out_list_junk, perm_out_list = locc_povm_func(gamma_ideal,
                                                             ips)

#print(perm_out_list[2])

syn_circuit = compile(perm_out_list[0])#, max_synthesis_size = 2)

syn_circuit.compress()
gate_dict = syn_circuit.gate_counts
#print(gate_dict.values())
cnot_count = list(gate_dict.values())[1]
print(cnot_count)
#print(syn_circuit.gate_counts)
#print(syn_circuit.num_cycles)
#print(syn_circuit.depth)
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

"""
#sp.synthesize(self, utry =I, data =0)


#num_qubits = 3
#qc = QuantumCircuit(num_qubits)
#qc.unitary(perm_out_list[0], qubits=range(num_qubits))
#qc.draw('mpl')
