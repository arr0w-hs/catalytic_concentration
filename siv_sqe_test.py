#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 18:58:16 2024

@author: hsharma4
"""

import numpy as np
import scipy as sc
import math as math
import pandas as pd
import matplotlib.pyplot as plt
from qutip import *
from qutip.measurement import measure, measurement_statistics, measure_observable

import sys
import os
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)
from siv_catalytic_transform_base import *
from optimized_distillation import *
from siv_state_prep import *

g = 10
gm0 = 0.1
gm1 = 0.1
delta = 10
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
num_reset = 1


raw_fid_list = []
cat_fidelity = []
cat_fid = []
gain_list = []
cat_state = []
x = []
kap = []
cat_prob = []
 
nocat_fid = []
nocat_prob = []
dist_fid_list = []
dist_prob_list = []
fid_ratio = []
prob_ratio = []
dames_fid_list = []
dames_prob_list = []

fip_cat = []
fip_nocat = []
fip_dist = []
fip_dames = []

"""preparing the bell states"""
for i in range(100):
    #print(i)
    kappa = 25#*(i+1)
    sqe_rate = (i)*0.01*np.pi
    
    
    #kappa = 3*((i)%20)+30
    #cnot_errore = [i/20*0.05, i/20*0.05, i/20*0.05, i/20*0.05]
    #sqe_rate = 0.001*i#i/20*0.05
    fip_dames.append(sqe_rate)
    
    kap.append(kappa)    
    param = [kappa,0,g,gm0,gm1,delta,0]
    rr = r_vector(param, param, loss_coeff)
    #print("rr", rr)
    lvec = l_vector(param, param, loss_coeff)
    #print(lvec)
    prob_noloss, prob_loss, total_prob = prepare_state_dm_test(psn_dm, cnot_errore, rr, lvec, num_reset, sqe_rate)
    print(prob_noloss)
    #prob_loss = sqe_rate*np.log(sqe_rate)+ (1-sqe_rate)*np.log((1-sqe_rate))
    fip_cat.append(prob_noloss)
    fip_nocat.append(prob_loss)
    fip_dist.append(total_prob)
    
    
    
plt.figure()
plt.plot(fip_dames, fip_cat)
plt.title("noloss prob")
plt.figure()
plt.plot(fip_dames, fip_nocat)
plt.figure()
plt.title("total prob")
plt.plot(fip_dames, fip_dist)