#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 28 10:56:40 2024

@author: hsharma4

for plotting fidelity vs measurement results
"""
import sys
import os

import pickle
import matplotlib.pyplot as plt

from qutip import *
from qutip.measurement import measure, measurement_statistics, measure_observable
from pathlib import Path
from base_siv_catalytic_transform import *
from base_siv_state_prep import prepare_dm_withreset, l_vector, r_vector
from base_distillation import distillation, dejmps

import numpy as np
import pandas as pd
plt.rcParams.update({'font.size': 12})
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)


g = 8.38
gamma_0 = 0.123
gamma_1 = 0.123
delta = 10
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

single_err = 0
sqe_error = [0,0,single_err]

cat_fidelity = []
gain_list = []
cat_state = []
x = []
kap = []

mea_list = []

fid_raw_list = []


n = 5000
flag = 0

ideal_state, ideal_state_loss, prob_ideal, prob_loss, mea_list_ideal = prepare_dm_withreset(psn_dm, cnot_errore_ideal,
                                      rr_ideal, lvec_ideal, 0, [0, 0, 0], 0)

"""preparing the bell states"""
for i in range(n):
    print(i)
    dist = 0

    final_state, final_state_loss, prob_final_state, prob_final_loss, mea_value = prepare_dm_withreset(
            psn_dm, cnot_errore, rr, lvec, num_reset, sqe_error, dist)

    raw_fid = np.sqrt(fidelity(ideal_state, final_state))
    fid_raw_list.append((raw_fid))
    mea_list.append(mea_value)

plt.figure()
plt.grid()
plt.scatter(mea_list, fid_raw_list, s = 5, c = "blue")


plt.figure()
plt.grid()
plt.hist(mea_list, bins = 64)
