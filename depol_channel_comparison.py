#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:23:30 2024

@author: hsharma4

for creating simulation data for comparison between 
(non-) catalytic locc and distillation in presence of
depolarising noise
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
from base_depol_channels import *
import numpy as np
import pandas as pd


plt.rcParams.update({'font.size': 12})

ph = basis(2, 0)
spin = basis(2,0)
nu = basis(2,0)
psnn = tensor(ph, spin, nu, spin, nu)
psn_dm = ket2dm(psnn)

alpha_list = []
prob_in_state_list = []

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

n = 100

"""preparing the bell states"""
for i in range(n):
    print(i)
    
    prob_in_state = 0.7-0.1*i/n #0.95
    alpha = 0.85# - i/n*0.49
    
    #print(prob_in_state)
    #final_state = r_state(alpha, prob_in_state)
    #final_state = new_state_depol(alpha, prob_in_state)
    final_state = new_state_pauli_z(alpha, prob_in_state)
    #final_state = new_state_pauli_x(alpha, prob_in_state)
    
    ideal_state = new_state_depol(1, 1)
    
    swc, afadf, aadsg = schmidt_decomp_of_dm(final_state)
    ops = [0.5, 0.5]
    #print(swc, "swc")
    
    fid_cat, prob_cat, post_locc_state, carbon_cat_st = catalytic_conversion(final_state)
    fid_nocat, prob_nocat, post_locc_state_nocat = non_catalytic_conversion(final_state)
    fid_dist, prob_dist = distillation(final_state, ideal_state, 0)
    fid_dames, prob_dames = dejmps(final_state, 0)
    
    cat_post = post_locc_state.ptrace([2,5])
    #print(post_locc_state.ptrace([1,4]))
    #print(post_locc_state.ptrace([2,5]))
    #print(post_locc_state.ptrace([3,6]), "post locc qobj")
    fid_reuse, prob_reuse, out_state_reuse, flag = catalytic_conversion_reuse(final_state, cat_post)
    #print(fid_cat, "fid cat")
    #print(fid_reuse, "fid reuse")

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
    fid_raw_list.append(1-(raw_fid))
    
    #print(raw_fid)
    #print((np.sqrt(alpha)+np.sqrt(1-alpha))**2/2)
    #print(fid_raw_one, fid_raw_two)

    prob_in_state_list.append(1-prob_in_state)
    alpha_list.append(1-alpha)
    #print(sdfaf)
    
#plt.figure()
#plt.grid()
#plt.scatter(alpha_list, fid_raw_list, s = 5, c = "blue")


data_dict = {
    "alpha_list": alpha_list,
    "prob_in_state_list": prob_in_state_list,
    "fid_raw_list": fid_raw_list,
    "fid_cat_list": fid_cat_list,
    "fid_nocat_list": fid_nocat_list,
    "fid_dist_list": fid_dist_list,
    "fid_cat_reuse_list": fid_cat_reuse_list,
    "prob_cat_list": prob_cat_list,
    "prob_nocat_list": prob_nocat_list,
    "prob_dist_list": prob_dist_list,
    "prob_cat_reuse_list": prob_cat_reuse_list,
    
    }
ts = pd.Timestamp.today(tz = 'Europe/Stockholm')
date_str = str(ts.date())

time_str = ts.time()
time_str = str(time_str.hour)+ str(time_str.minute) + str(time_str.second)
print(time_str)
data_directory = os.path.join(dir_name+"/data", date_str+"_depol_channel_comparison/")
plots_directory = os.path.join(dir_name+"/plots", date_str+"_depol_channel_comparison/")

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


with open(data_directory+ time_str +'.pkl', 'wb') as f:  # open a text file
    pickle.dump(data_dict, f)


with open(__file__) as f:
    data = f.read()
    f.close()

metadata_dict = {
    "num of total points n": n,
    "alpha_max": np.max(alpha_list),
    "alpha_min": np.min(alpha_list),
    "prob_in_state_max": np.max(prob_in_state_list),
    "prob_in_state_min": np.min(prob_in_state_list),
    "depol_channel": "x"
    }

with open(data_directory + time_str +'_metadata.txt', mode="w") as f:
    f.write(str(metadata_dict))
    f.close()
