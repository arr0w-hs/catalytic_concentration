#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 13:37:54 2024

@author: hsharma4
for analysing degradation of catalyst state
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

import numpy as np
import scipy as sc
import math as math
import pandas as pd
import pickle
import matplotlib.pyplot as plt

from qutip import *
from qutip.measurement import measure, measurement_statistics, measure_observable
from pathlib import Path

from base_siv_catalytic_transform import *
from base_siv_state_prep import *
from base_distillation import *

plt.rcParams.update({'font.size': 12})
g = 8.38
gm0 = 0.123
gm1 = 0.123
delta = 24
kappa = 21.8
loss_coeff = 0.01
param = [kappa,0,g,gm0,gm1,delta,0]


cnot_errore = [0, 0, 0, 0]
rr = r_vector(param, param, loss_coeff)
lvec = l_vector(param, param, loss_coeff)

num_reset = 2

ph = basis(2, 0)
spin = basis(2,0)
nu = basis(2,0)
psnn = tensor(ph, spin, nu, spin, nu)
psn_dm = ket2dm(psnn)

cnot_errore_ideal = [0, 0, 0, 0]
rr_ideal = [1,1,-1,-1]
lvec_ideal = [0,0,0,0]

sqe_list = []
cnot_err_list = []
loss_coeff_list = []

cat_fidelity = []
gain_list = []
cat_state = []
x = []
kap = []

fid_raw_list = []
fid_nocat_list = []
fid_dist_list = []
fid_dames_list = []
fid_cat_list = []
fid_of_catalyst = []

prob_nocat_list = []
prob_dist_list = []
prob_dames_list = []
prob_cat_list = []

prob_cl= []
fid_cl = []
fid_cat_cl = []

fid_ratio = []
prob_ratio = []

fip_cat_list = []
fip_nocat_list = []
fip_dist_list = []
fip_dames_list = []

prob_coeff = []
n = 1000
repeat = 100
count = 1
flag = 0
"""preparing the bell states"""
for ii in range(20):
    delta = 2*(ii+5)
    print(delta)
    for i in range(n):
        fid_cat_list = []
        prob_cat_list = []
        #print(i)
        #i = i%repeat
    
        """
        #kappa = ((i)%80)+10
        #kappa = 100
        #kappa = ((i)%150)+65
        """
    
        cnot_err = 0.01#.001*i
        cnot_errore = [cnot_err, cnot_err, cnot_err, cnot_err]
    
        single_err = 0.01#0.001*i
        sqe_error = [single_err/3, single_err/3, single_err/3]
    
        #loss_coeff = 0.0005*i
    
        param = [kappa,0,g,gm0,gm1,delta,0]
        rr = r_vector(param, param, loss_coeff)
        lvec = l_vector(param, param, loss_coeff)
    
        #final_state, final_state_loss, prob_final_state, prob_final_loss, mea_value = prepare_dm_withreset(psn_dm, cnot_errore, rr, lvec, num_reset, sqe_error,0)
    
        if mea_value != 15:
            continue
    
        ideal_state, ideal_state_loss, prob_ideal, prob_loss, mea_list_ideal = prepare_dm_withreset(psn_dm, cnot_errore_ideal,
                                                                                          rr_ideal, lvec_ideal, 0, [0, 0, 0],0)
        
        bell_st = 1/np.sqrt(2)*(tensor(basis(2,0), basis(2,0)) + tensor(basis(2,1), basis(2,1)))
        #print(fidelity(bell_st, final_state.ptrace([1,3])))
        #print(fidelity(bell_st, final_state.ptrace([2,4])))
        fid_cat, prob_cat, post_locc_state, carbon_cat_st = catalytic_conversion(final_state)
    
        fid_nocat, prob_nocat, post_locc_state_nocat = non_catalytic_conversion(final_state)
        fid_dist, prob_dist = distillation(final_state, ideal_state, 0)
        #print(fid_cat, prob_cat)
        #print(fid_nocat, prob_nocat)
    
        cat_post = post_locc_state.ptrace([3,6])
        output_state_post = post_locc_state.ptrace([0,1,2,4,5])
        cat_state.append((carbon_cat_st[0]))
        raw_fid = np.sqrt(fidelity(ideal_state, final_state))
    
        fid_cat_list.append(fid_cat)
        prob_cat_list.append(prob_cat)
        fid_of_catalyst.append(fidelity(carbon_cat_st, cat_post))
        print("fidelity of catalyst after one use", fidelity(carbon_cat_st, cat_post))
    
        #psnc_dm = tensor(cat_post, final_state)
        #psnc_dm = psnc_dm.permute([2,3,4,0,5,6,1])
        #print("dfadf")
        #print(final_state.dims)
        #print(cat_post.dims)
        count = 0
        while count <25:#pro > pro_nocat and
            #print(count)
            fid_reuse, prob_reuse, output_density_mat, flag = catalytic_conversion_reuse(final_state, cat_post)
            cat_post = output_density_mat.ptrace([3,6])
            #print(cat_post[0,0])
            if flag == 0:
                fid_cat_list.append(fid_reuse)
                #fid_nocat_list.append(fid_nocat)
                #print(pro_reuse)
                prob_cat_list.append(prob_reuse)
                #prob_nocat_list.append(pro_nocat)
                fid_of_catalyst.append(fidelity(carbon_cat_st, cat_post))
                pro = prob_reuse
                count += 1
            else:
                print("flag is 1")
                count = 50
        #print(cat_post)
        #print(carbon_cat_st)
        print("fidelity of catalyst after last use", fidelity(carbon_cat_st, cat_post))
        #print(fid, pro)
        #print(count)
        if mea_value == 15:
           break
    #print(ii)
    fid_cl.append(fid_cat_list)
    prob_cl.append(prob_cat_list)
    fid_cat_cl.append(fid_of_catalyst)


#print(count)
print(len(fid_cat_list))
fid_cl = np.reshape(fid_cl, [20,26])
prob_cl = np.reshape(prob_cl, [20,26])

plt.figure()
plt.imshow(fid_cl, aspect = 'auto', interpolation='nearest', 
           vmin=None, vmax=None)#,cmap='hot'
plt.colorbar() 
plt.title("Fidelity")
plt.ylabel('Delta value')
plt.xlabel("Number of reuse")

plt.figure()
plt.imshow(prob_cl, aspect = 'auto', interpolation='nearest', 
           vmin=None, vmax=None)#,cmap='hot'
plt.colorbar() 
plt.title("Probability")
plt.ylabel('Delta value')
plt.xlabel("Number of reuse")

"""
x1 = np.linspace(1, count+1, len(fid_cat_list))
plt.figure()
plt.grid()
plt.plot(x1, fid_cat_list)
plt.axhline(y=fid_nocat, linewidth=0.7)
plt.axhline(y=fid_dist, linewidth=0.7)
#plt.yscale("log")
plt.ylabel('Fidelity of transformation')
plt.xlabel('count')
plt.xticks(rotation=45)
plt.savefig(dir_name+"mid_fid_fid.png", dpi=1000, format="png")

plt.figure()
plt.grid()
plt.plot(x1, prob_cat_list)#[0:-1]
plt.axhline(y=prob_nocat, linewidth=0.7)
plt.axhline(y=prob_dist, linewidth=0.7)
#plt.yscale("log")
plt.ylabel('probability')
plt.xlabel('count')
plt.xticks(rotation=45)
plt.savefig(dir_name+"mid_fid_prob.png", dpi=1000, format="png")

plt.figure()
plt.grid()
plt.plot(x1, fid_of_catalyst)
#plt.yscale("log")
plt.ylabel('Fidelity of catalyst')
plt.xlabel('count')
plt.xticks(rotation=45)
plt.savefig(dir_name+"mid_fid_fidofcat.png", dpi=1000, format="png")
"""















