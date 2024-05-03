#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 14:42:07 2024

@author: hsharma4

gain as we vary the dimesnions of the catalyst
"""

import numpy as np
import scipy as sc
import math as math
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
import seaborn as sns
from qutip import *
from qutip.measurement import measure, measurement_statistics, measure_observable

import sys
import os
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)
from base_catalyst import *
from base_slocc import *

res_dif_ini_state2 = []
dc = 2
k = 2

    
for i in range(3):
    k = 2**(i+1)
    
    cat_guess = np.random.randint(1, 100000, size=dc)
    cat_guess = cat_guess/np.linalg.norm(cat_guess, ord=1)
    
    res_ca = []
    for j in range(110, 220):
        pro = 1
        flag = 'fail'
        flag_count = 0
        input_state = [j/220, 1-j/220]
        output_state = [0.5, 0.5]
        result = catalytic_concentration2(output_state, input_state, cat_guess, k, dc)
        while flag == 'fail':
            result = catalytic_concentration2(output_state, input_state, cat_guess, k, dc)
            flag = result[4]
            flag_count += 1
            if flag_count == 10:
                flag = 'success'
        pro = result[1]
        cat_guess = result[3]
        
        res_ca.append(pro)
    
    res_dif_ini_state2.append(res_ca)
    
plt.figure()
a = np.linspace(0.5, 1, 110)

plt.plot(a, res_dif_ini_state2[0])
plt.plot(a, res_dif_ini_state2[1])
plt.plot(a, res_dif_ini_state2[2])
#plt.plot(a, res_dif_ini_state_nc[0])
#plt.plot(a, res_dif_ini_state_nc[1])
#plt.plot(a, res_dif_ini_state_nc[2])

plt.grid()
#plt.title(f"Performance with %d-tensored initial state" % k)
plt.xlabel(r'Initial state $\alpha$')
#plt.ylabel('Probability of transformation')
plt.ylabel(r'Catalytic gain ($\eta$)')
plt.legend(['2 copies', '4 copies', '8 copies'])#, 'k=4'])
plt.savefig(dir_name+"/catalytic_gain_varying_ini_state"+".png", dpi=1000, format="png")





