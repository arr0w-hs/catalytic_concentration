#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 27 10:41:58 2024

@author: hsharma4
"""

import numpy as np
import scipy as sc
import math as math
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from qutip import *
from qutip.measurement import measure, measurement_statistics, measure_observable

import sys
import os
sys.path.append(os.path.dirname(__file__))
from base_siv_catalytic_transform import *
from base_distillation import *

g = 8.38
gm0 = 0.123
gm1 = 0.123
delta = 248
kappa = 10
loss_coeff = 0.01
param = [kappa,0,g,gm0,gm1,delta,0]

ph = basis(2, 0)
spin = basis(2,0)
nu = basis(2,0)
psnn = tensor(ph, spin, nu, spin, nu)
psn_dm = ket2dm(psnn)

cnot_errore_ideal = [0, 0, 0, 0]
rr_ideal = [1,1,-1,-1]
lvec_ideal = [0,0,0,0]

cnot_errore = [0, 0, 0, 0]
rr = r_vector(param, param, loss_coeff)
lvec = l_vector(param, param, loss_coeff)
sqe_rate = 0

cnot_err = 0#.001*i
cnot_errore = [cnot_err, cnot_err, cnot_err, cnot_err]

single_err = 0#0.001*i
sqe_rate = [single_err/3, single_err/3, single_err/3]

psn_noloss, psn_loss, measure_output = entangle_spins_dm(psn_dm, rr, lvec, sqe_rate)

psn_state, prob = closest_pure_state(psn_noloss)
psn_sch_bas, psn_transform, ss = basis2schmidt(psn_state)
#print(psn_state)
#print(psn_sch_bas)
"""
    function for optimizing both alice's and bob's spin states
    both will be different for maximum increase in circuit performance
"""

def func_fidelity(theta, phi, out_state):
    
    psn_noloss, psn_loss, measure_output = entangle_spins_dm(psn_dm, rr, lvec, sqe_rate)
    psn_ideal, psn_null, measure_output = entangle_spins_dm(psn_dm, rr_ideal, lvec_ideal, 0)
    
    f = -1*fidelity(psn_noloss, psn_ideal)
    
    
    if np.linalg.norm(cat_1, ord=1) != 0:
        cat_1 = cat_1/np.linalg.norm(cat_1, ord=1)

    in_1 = np.reshape(np.tensordot(in_state, cat_1, 0), np.shape(in_state)[0]*np.shape(cat_1)[0])
    out_1 = np.reshape(np.tensordot(out_state, cat_1, 0), np.shape(out_state)[0]*np.shape(cat_1)[0])
    in_1 = np.sort(in_1)[::-1]
    out_1 = np.sort(out_1)[::-1]

    if prob_of_transformation2(out_state, in_state) !=0:
        f = -1*prob_of_transformation2(out_1, in_1)
    else:
        f = 0
    
    return f

#instate is input, outstate is output state, 
#num_k is the number of input states to be tensored
#d_c is the dimension of the catalyst
def catalytic_concentration(outstate, instate, num_k, d_c):
    instate = instate/np.linalg.norm(instate, ord=1)
    instate_tensored = self_tensor_prod(instate, num_k)

    bnds = []
    for ii in range(d_c):
        bnds.append((0, 1))
    cat_guess = np.random.randint(1, 100000, size=d_c)
    cat_guess = cat_guess/np.linalg.norm(cat_guess, ord=1)
        
    res = sc.optimize.minimize(func_prob, cat_guess, args = (instate_tensored, outstate), 
                               method='SLSQP', bounds = bnds)
    
    
    cat_final = res.x
    if np.linalg.norm(cat_final, ord=1) == 0:
        flg = "fail"
        print("fail encountered")
    else:
        cat_final = cat_final/np.linalg.norm(cat_final, ord=1)
        flg = 'success'
    cat_final = np.sort(cat_final)[::-1]
    
    pcr = -1*res.fun
    pncr = prob_of_transformation2(outstate, instate_tensored)
    gain = pcr/pncr
    
    if gain >= 0.9999 and gain <= 1.0001:
        cat_final = [0.51, 0.49]
        
    if gain > 50:
        print("gain > 50")
        print(cat_final, "catalyst state")
        print(instate_tensored, "initial state ternsored")
    return pncr, gain, pcr, cat_final, flg



"""preparing the bell states"""
for i in range(500):
    #print(i)
    #kappa = 250
    kappa = 21.8
    #kappa = ((i)%150)+70
    #kappa = ((i)%70)+40
    #cnot_errore = [i/20*0.05, i/20*0.05, i/20*0.05, i/20*0.05]
    
    #sqe_rate = 0.01*i
    
    
    
    kap.append(kappa)    
    param = [kappa,0,g,gm0,gm1,delta,0]
    rr = r_vector(param, param, loss_coeff)
    lvec = l_vector(param, param, loss_coeff)
    #print(lvec)
    final_state, final_state_loss, prob_final_state, prob_final_loss, mea = prepare_dm_withreset(psn_dm, cnot_errore, rr, lvec, num_reset, sqe_rate)
    #print(mea_list_obtained)
    #mea = np.flip(np.reshape(np.asarray(mea_list_obtained), 2*len(mea_list_obtained)))
    #mea = int(''.join(map(lambda mea: str(int(mea)), mea)), 2)
    
    #for j in range(len(mea_list_obtained)):
    #    mm = mea_list_obtained[j]
    
    #if mea != 5:
    #    continue
    
    #print(mea)
    print(i)
    #mea_list.append(mea)
    
    #ideal_state, ideal_state_loss, prob_ideal, prob_loss, mea_list_ideal = prepare_state_dm_withreset(psn_dm, cnot_errore_ideal, 
                                                                                      #rr_ideal, lvec_ideal, 0, 0)
    #swc, afadf, aadsg = schmidt_decomp_of_dm(final_state)
    #swc_ideal, adfa, adfadsfasdfasfasf = schmidt_decomp_of_dm(ideal_state)
    #print(np.sum(np.sqrt(swc)*np.sqrt(swc_ideal)))
    #print(fidelity(final_state, ideal_state))
    
    #fid, pro, post_locc_state, carbon_cat_st, catalyst_gain = catalytic_conversion(final_state)
    #fid_nocat, pro_nocat, post_locc_state_nocat = non_catalytic_conversion(final_state)
    #fid_dist, prob_dist = distillation(final_state, ideal_state, 0)
    #fid_dames, prob_dames = dejmps(final_state, 0)
    #fid_dist, prob_dist = bbpsw(final_state, 0)
    
    #fid_cat.append(fid)
    #fid_no_cat.append(fid_nocat)
    
    
    #cat_prob.append(pro)
    #nocat_prob.append(pro_nocat)
    
    
    #fip_cat.append(fid*pro)
    #fip_nocat.append((fid_nocat*pro_nocat))
    
    
    #cat_post = post_locc_state.ptrace([3,6])
    #cat_state.append((carbon_cat_st[0]))
    #raw_fid = np.sqrt(fidelity(ideal_state, final_state))
    
    #print(raw_fid)
    #cat_fid_post = fidelity(carbon_cat_st, cat_post)
    
    #cat_fidelity.append(cat_fid_post)
    #x.append(i)
    #raw_fid_list.append((raw_fid))
    #gain_list.append(catalyst_gain)
    
print(len(fid_cat))    

    #fid_ratio.append(fid/fid_nocat)
    #prob_ratio.append(pro/pro_nocat)
    
    #print(mea_list_obtained, raw_fid)
    
#with open('fid_cat.pickle', 'wb') as handle:
#    pickle.dump(fid_cat, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
#plt.figure()    
#plt.scatter(raw_fid_list, cat_fidelity)
#plt.title('Catalyst fidelity')
#plt.ylabel('Fidelity of catalyst after one use')
#plt.savefig(dir_name+"/fidelity"+".png", dpi=1000, format="png")

plt.figure()
plt.hist(mea_list)

