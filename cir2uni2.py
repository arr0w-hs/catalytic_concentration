#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  6 13:09:23 2024

@author: hsharma4
for a=0.9, p =0.95
"""

import sys
import os

import pickle
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

from qutip import *
from qutip.measurement import measure, measurement_statistics, measure_observable
from pathlib import Path
from base_siv_catalytic_transform import *
from base_siv_state_prep import prepare_dm_withreset, l_vector, r_vector
from base_distillation import distillation, dejmps, distillation_operation
from syn2depol import apply_locc_conversion, apply_schmidt_conversion, add_aux, apply_slocc_conversion,to_be_syn_nc, measure_aux
from base_depol_channels import new_state_depol, new_state_pauli_z, new_state_pauli_x1
from bqskit.ir.circuit import Circuit
from qutip_qip.circuit import QubitCircuit
from syn2depol import extend_perm
import numpy as np
import pandas as pd
import time
#import rsmf
#print(dir_name+"/quantum-template.tex")
# Get formatter specifications from tex file
#fmt = rsmf.setup(dir_name+"/quantum-template.tex")

plt.rcParams.update({'font.size': 12})

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

    #a = schmidt_decomp_of_dm(output_state)
    #output_state = (a[1])

    fid = fidelity(output_state, output_state_qobj)

    return fid, prob, output_state



#data_location = '/145345.pkl'   #2024-08-01_cat_disti_comparison_nc
data_location = '/164828.pkl'
with open(dir_name+'/test_circuits/2024-09-06_cat_disti_comparison_nc'+data_location, 'rb') as f:
    data_dict_loaded = pickle.load(f)
    f.close()

#print(data_dict_loaded.keys())
list_circ_nc = data_dict_loaded["list_circs"]
#print(data_dict_loaded["a"], data_dict_loaded["p"])



data_dir = os.path.join(dir_name, "cirl3")#cirl is for 0.95, cirl_1 is for 0.85


#data_dir = os.path.join(dir_name, "cirl_1")
#plot_dir = os.path.join(dir_name, "er_results_data/plots")
#df_list = []

for root, _, files in os.walk(data_dir):
    for file in files:
        if file == '.DS_Store':
            continue
        #print(file)
        with open(os.path.join(data_dir, file), 'rb') as f:
            data_dict_loaded = pickle.load(f)
            f.close()

        out_df = (data_dict_loaded["list_circs"])
        #print(len(out_df))



df_list = []


pcl = '/15318.pkl'
with open(data_dir+pcl, 'rb') as f:
    data_dict_loaded = pickle.load(f)
    f.close()
cir0 = data_dict_loaded["list_circs"][0]

pcl = '/165344.pkl'          #2 elements
with open(data_dir+pcl, 'rb') as f:
    data_dict_loaded = pickle.load(f)
    f.close()
cir1 = data_dict_loaded["list_circs"]


pcl = '/155016.pkl'
with open(data_dir+pcl, 'rb') as f:
    data_dict_loaded = pickle.load(f)
    f.close()
cir2 = data_dict_loaded["list_circs"][0]

pcl = '/165637.pkl'           #2 elements
with open(data_dir+pcl, 'rb') as f:
    data_dict_loaded = pickle.load(f)
    f.close()
cir3 = data_dict_loaded["list_circs"]

pcl = '/165856.pkl'        #2 elements
with open(data_dir+pcl, 'rb') as f:
    data_dict_loaded = pickle.load(f)
    f.close()
cir4 = data_dict_loaded["list_circs"]

pcl = '/142639.pkl'
with open(data_dir+pcl, 'rb') as f:
    data_dict_loaded = pickle.load(f)
    f.close()
cir5 = data_dict_loaded["list_circs"][0]

pcl = '/164626.pkl'
with open(data_dir+pcl, 'rb') as f:
    data_dict_loaded = pickle.load(f)
    f.close()
perm = data_dict_loaded["list_circs"]

pcl = '/1845.pkl'
with open(data_dir+pcl, 'rb') as f:
    data_dict_loaded = pickle.load(f)
    f.close()                               #u and v are the first two elements
cir_slocc = data_dict_loaded["list_circs"]  #slocc povm is the third element

#print(len(cir0[0][0][0]), "cir0")

#print(len(cir_slocc[2]))

#print(len(cir1[0]))

cir_temp = Circuit(cir1[0].num_qudits)
for ele in cir1:
    for elem in ele:
        cir_temp.append(elem)
cir1 = cir_temp

cir_temp = Circuit(cir3[0].num_qudits)
for ele in cir3:
    for elem in ele:
        cir_temp.append(elem)
cir3 = cir_temp

cir_temp = Circuit(cir4[0].num_qudits)
for ele in cir4:
    for elem in ele:
        cir_temp.append(elem)
cir4 = cir_temp

#print(len(cir1))
#print((cir1[0]))
#print(len())


#qc = QubitCircuit(N=3, num_cbits=3)


#u = list_unitary[2]
#v = list_unitary[3]
#locc_uni = list_unitary[0]

#print(locc_uni)
#print(v)
#aa = tensor(I, u, v)
#aa.dims = [[2]*5, [2]*5]
#print(aa)

perm1 = perm[0][0][1]
perm3 = perm[0][1][1]
perm4 = perm[0][2][1]

cir1list = []
cir1list.append(cir1)
cir1list.append(perm1)

cir3list = []
cir3list.append(cir3)
cir3list.append(perm3)

cir4list = []
cir4list.append(cir4)
cir4list.append(perm4)
#cir2.append(perm2)
#cir4.append(perm4)

locc_cir = [cir0[0], cir1list, cir2[0], cir3list, cir4list, cir5[0]]
cat = [locc_cir, cir_slocc[1], cir_slocc[2], cir_slocc[2]]
#print(u_op.gate_counts)

# print(len(cat[0]))
# for ele in list_circ_nc[0]:
#     #print(1)
#     for elem in ele:
#         #print(elem)
#         print(type(elem))
#         #print(type[elem[0]], type(elem[1]), "elem")
#         #for el in elem:
#         #    print(type(el), "el")
# print("adfasfda")
# for ele in cat[0]:
#     #print(1)
#     for elem in ele:
#         #print(elem)
#         print(type(elem))
#         #print(type[elem[0]], type(elem[1]), "elem")
#         #for el in elem:
#         #    print(type(el), "el")

a = 0.9
p = 0.95
final_state = new_state_pauli_x1(a, p)
ideal_state = new_state_pauli_x1(1, 1)

fid_dist, prob, ops = distillation(final_state, ideal_state, 0)
print(fid_dist, prob, "distillation")
print(ops)
fid_nocat, prob_nocat, post_locc_state_nocat = oper_err(list_circ_nc, final_state, 0, 0, 0)
print(fid_nocat, prob_nocat, "slocc")

#fid_cat, prob_cat, post_locc_state_cat = oper_err(cat, final_state, 1, 0, 0)
#print(fid_cat, prob_cat, "cat_slocc")

#print(hh)

x = []
fid_list = []
prob_list = []

fid_cat_list = []
prob_cat_list = []

fid_dist_list = []
prob_dist_list = []


#print(output_state_qobj)
for i in range(20):
    print(i)
    t1 = time.time()
    err = (i)*0.0001
    cerr = 0
    x.append(err)
    fid_nocat, prob_nocat, post_locc_state_nocat = oper_err(list_circ_nc, final_state, 0, err, cerr)

    hh = distillation_operation(final_state, ops, err, cerr)


    fid_dist_list.append(hh[0])
    prob_dist_list.append(hh[1])

    fid_list.append(fid_nocat)
    prob_list.append(prob_nocat)

    fid_cat, prob_cat, post_locc_state_cat = oper_err(cat, final_state, 1, err, cerr)


    fid_cat_list.append(fid_cat)
    #print(fid_cat)
    prob_cat_list.append(prob_cat)

    #print(i, time.time()-t1)


#fig = fmt.figure()
plt.figure()
plt.plot(x, fid_cat_list, label = "CEC")
plt.plot(x, fid_list, label = "SEC")
plt.plot(x, fid_dist_list, label = "Distillation")
#plt.yscale("log")
plt.legend()
plt.ylabel('Fidelity')
plt.xlabel('Error rate')
plt.xticks(rotation=45)
plt.grid()
plt.savefig(dir_name+"/_fidelity_cat_sqe11.png", dpi=1000, format="png", bbox_inches = 'tight')
plt.savefig(dir_name+"/_fidelity_cat_sqe11.pdf", dpi=1000, format="pdf", bbox_inches = 'tight')
#plt.show()

plt.figure()
plt.grid()
plt.plot(x, prob_cat_list, label = "CEC")
plt.plot(x, prob_list, label = "SEC")
plt.plot(x, prob_dist_list, label = "Distillation")
#plt.yscale("log")
plt.legend()
plt.ylabel('Probability of success')
plt.xlabel('Error rate')
plt.xticks(rotation=45)
plt.savefig(dir_name+"/_probability_cat_sqe11" + ".png", dpi=1000, format="png", bbox_inches = 'tight')
plt.savefig(dir_name+"/_probability_cat_sqe11" + ".pdf", dpi=1000, format="pdf", bbox_inches = 'tight')
#plt.show()
print(1)
