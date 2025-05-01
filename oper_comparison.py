#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 30 09:56:02 2025

@author: hsharma4
"""
import sys
import os
#from pathlib import Path
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

from base_distillation import distillation, distillation_operation
from base_depol_channels import new_state_pauli_x1#new_state_depol, new_state_pauli_z,
from base_oper_error import oper_err



x = []
fid_list = []
prob_list = []
fid_cat_list = []
prob_cat_list = []
fid_dist_list = []
prob_dist_list = []


a = 0.85
p = 0.95

final_state = new_state_pauli_x1(a, p)
fid_dist, prob, ops = distillation(final_state, 0)

for i in range(20):

    err = 1.2**(i)*0.0001
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


fs = 15
plt.figure()
plt.plot(x, fid_cat_list, 'o-', label = "CEC")
plt.plot(x, fid_list, '.-', label = "NEC")
plt.plot(x, fid_dist_list, 'v-', label = "Distillation")
# plt.yscale("log")
plt.xscale("log")
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
plt.plot(x, prob_list, '.-', label = "NEC")
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
