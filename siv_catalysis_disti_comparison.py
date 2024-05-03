#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 11:03:48 2023

@author: hsharma4
for creating simulation data for comparison between 
(non-) catalytic locc and distillation
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
from base_distillation import distillation, dejmps

import numpy as np
import pandas as pd


plt.rcParams.update({'font.size': 12})





g = 8.38
gamma_0 = 0.123
gamma_1 = 0.123
delta = 24
kappa = 21.8
loss_coeff = 0.05
param = [kappa,0,g,gamma_0,gamma_1,delta,0]


cnot_errore = [0, 0, 0, 0]
rr = r_vector(param, param, loss_coeff)
lvec = l_vector(param, param, loss_coeff)

num_reset = 2

ph = basis(2, 0)
spin = basis(2,0)
nu = basis(2,0)
psnn = tensor(ph, spin, nu, spin, nu)
psn_dm = ket2dm(psnn)

dist = 0
cnot_errore_ideal = [0, 0, 0, 0]
rr_ideal = [1,1,-1,-1]
lvec_ideal = [0,0,0,0]

sqe_list = []
cnot_err_list = []
loss_coeff_list = []
distance_list = []
delta_list = []

cat_fidelity = []
gain_list = []
cat_state = []
x = []
kap = []

fid_raw_list = []
fid_raw_one = []
fid_raw_two = []
fid_nocat_list = []
fid_dist_list = []
fid_dames_list = []
fid_cat_list = []
fid_cat_reuse_list = []

prob_nocat_list = []
prob_dist_list = []
prob_dames_list = []
prob_cat_list = []
prob_cat_reuse_list = []

fid_ratio = []
prob_ratio = []

fip_cat_list = []
fip_nocat_list = []
fip_dist_list = []
fip_dames_list = []
fip_cat_reuse_list = []

n = 2000
repeat = 50
flag = 0

"""preparing the bell states"""
for i in range(n):
    print(i)
    dist = 0
    i = i%repeat
    """
    #kappa = ((i)%80)+10
    #kappa = 100
    #kappa = ((i)%150)+65
    """
    #delta = 15+i
    cnot_err = 0.001*i
    cnot_errore = [cnot_err, cnot_err, cnot_err, cnot_err]

    single_err = 0.001*i
    sqe_error = [0,0,single_err]

    #loss_coeff = 0.0005*i

    param = [kappa,0,g,gamma_0,gamma_1,delta,0]
    rr = r_vector(param, param, loss_coeff)
    lvec = l_vector(param, param, loss_coeff)

    final_state, final_state_loss, prob_final_state, prob_final_loss, mea_value = prepare_dm_withreset(
            psn_dm, cnot_errore, rr, lvec, num_reset, sqe_error, dist)

    if mea_value != 15:
        continue

    ideal_state, ideal_state_loss, prob_ideal, prob_loss, mea_list_ideal = prepare_dm_withreset(psn_dm, cnot_errore_ideal,
                                          rr_ideal, lvec_ideal, 0, [0, 0, 0], 0)
    #swc, afadf, aadsg = schmidt_decomp_of_dm(final_state)
    #swc_ideal, adfa, adfadsfasdfasfasf = schmidt_decomp_of_dm(ideal_state)
    ops = [0.5, 0.5]
    
    

    fid_cat, prob_cat, post_locc_state, carbon_cat_st = catalytic_conversion(final_state)
    fid_nocat, prob_nocat, post_locc_state_nocat = non_catalytic_conversion(final_state)
    fid_dist, prob_dist = distillation(final_state, ideal_state, 0)
    fid_dames, prob_dames = dejmps(final_state, 0)
    
    cat_post = post_locc_state.ptrace([2,5])
    #print(post_locc_state.ptrace([1,4]))
    #print(post_locc_state.ptrace([2,5]))
    #print(post_locc_state.ptrace([3,6]), "post locc qobj")
    fid_reuse, prob_reuse, out_state_reuse, flag = catalytic_conversion_reuse(final_state, cat_post)


    fid_cat_list.append(1-fid_cat)
    fid_nocat_list.append(1-fid_nocat)
    fid_dist_list.append(1-fid_dist)
    fid_dames_list.append(fid_dames)
    fid_cat_reuse_list.append(1-fid_reuse)

    prob_cat_list.append(prob_cat)
    prob_nocat_list.append(prob_nocat)
    prob_dist_list.append(prob_dist)
    prob_dames_list.append(prob_dames)
    prob_cat_reuse_list.append(prob_reuse)
    
    fip_cat_list.append(fid_cat*prob_cat)
    fip_nocat_list.append((fid_nocat*prob_nocat))
    fip_dist_list.append(fid_dist*prob_dist)
    fip_dames_list.append(fid_dames*prob_dames)
    fip_cat_reuse_list.append(fid_reuse*prob_reuse)
    
    cat_state.append((carbon_cat_st[0]))
    cat_fid_post = fidelity(carbon_cat_st, cat_post)
    cat_fidelity.append(cat_fid_post)


    bell_st = 1/np.sqrt(2)*(tensor(basis(2,0), basis(2,0)) + tensor(basis(2,1), basis(2,1)))
    raw_fid = np.sqrt(fidelity(ideal_state, final_state))
    fid_raw_one.append(fidelity(bell_st, final_state.ptrace([1,3])))
    fid_raw_two.append(fidelity(bell_st, final_state.ptrace([2,4])))
    fid_raw_list.append((raw_fid))


    x.append(i)

    #gain_list.append(catalyst_gain)

    cnot_err_list.append(cnot_err)
    sqe_list.append(single_err)
    loss_coeff_list.append(loss_coeff)
    kap.append(kappa)
    distance_list.append(dist)
    delta_list.append(delta)

plt.figure()
plt.grid()
plt.scatter(loss_coeff_list, fid_raw_list, s = 5, c = "blue")

data_dict = {
    "param": param,
    "num_reset": num_reset,
    "cavity_loss": loss_coeff_list,
    "sqe_list": sqe_list,
    "cnot_error_list": cnot_err_list,
    "fid_raw_list": fid_raw_list,
    "fid_cat_list": fid_cat_list,
    "fid_nocat_list": fid_nocat_list,
    "fid_dist_list": fid_dist_list,
    "fid_cat_reuse_list": fid_cat_reuse_list,
    "prob_cat_list": prob_cat_list,
    "prob_nocat_list": prob_nocat_list,
    "prob_dist_list": prob_dist_list,
    "prob_cat_reuse_list": prob_cat_reuse_list,
    "distance_list": distance_list
    
    }
ts = pd.Timestamp.today(tz = 'Europe/Stockholm')
date_str = str(ts.date())

time_str = ts.time()
time_str = str(time_str.hour)+ str(time_str.minute) + str(time_str.second)
print(time_str)
data_directory = os.path.join(dir_name+"/data", date_str+"_cat_disti_comparison/")
plots_directory = os.path.join(dir_name+"/plots", date_str+"_cat_disti_comparison/")

date_folder = Path(data_directory)
if date_folder.exists():
    print("date folder exists")
else:
    os.mkdir(data_directory)

plot_folder = Path(plots_directory)
if plot_folder.exists():
    print("plot folder exists")
else:
    os.mkdir(plots_directory)

print(len(fid_raw_list))
with open(data_directory+ time_str +'.pkl', 'wb') as f:  # open a text file
    pickle.dump(data_dict, f)


with open(__file__) as f:
    data = f.read()
    f.close()

metadata_dict = {
    "param": param,
    "num_reset": num_reset,
    "sqe_max": np.max(sqe_list),
    "sqe_min": np.min(sqe_list),
    "cnot_err_max": np.max(cnot_err_list),
    "cnot_err_min": np.min(cnot_err_list),
    "cavity_loss_max": np.max(loss_coeff_list),
    "cavity_loss_min": np.min(loss_coeff_list),
    "distance_max": np.max(distance_list),
    "distance_min": np.min(distance_list),
    "num of total points n": n,
    "repetition": repeat,
    "delta_max": np.max(delta_list),
    "delta_min": np.min(delta_list),
    "only pauli error": "Z"
    }

with open(data_directory + time_str +'_metadata.txt', mode="w") as f:
    f.write(str(metadata_dict))
    f.close()
