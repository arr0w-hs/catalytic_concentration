#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 11 13:26:52 2023

@author: hsharma4
"""


import numpy as np
import scipy as sc
import math as math
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 12})
from qutip import *
from qutip.measurement import measure, measurement_statistics, measure_observable
import networkx as nx

import sys
import os
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

from base_slocc import *
from base_state_change import *
from base_catalyst import *

pro_cat = []
pro_nocat = []
x_lim = 0.501
y_lim = 0.501
n = 100
nx, ny = (n, n)
x = np.linspace(x_lim, 0.9999, nx)
y = np.linspace(y_lim, 0.9999, ny)

dc = 2
k = 2
cat_guess = np.random.randint(1, 100000, size=dc)
cat_guess = cat_guess/np.linalg.norm(cat_guess, ord=1)
dim_cat = []
cat_list = []
"""
ips = [0.75, 0.25]
for j in range(1):
    
    pro_cat = []
    dim_cat = []
    for i in range(30):
        dc = (i+1)
        x = 0.75+ 0.05*(j)
        ips = [x, 1-x]
        
        cat_guess = np.random.randint(1, 100000, size=dc)
        cat_guess = cat_guess/np.linalg.norm(cat_guess, ord=1)    
        #for i in range(nx):
        #ips = np.asarray([1.498-x[i], x[i]-0.498])
        #ips = np.asarray([x[i], 1-x[i]])
        #ips = np.asarray([1.451-x[i], x[i]-0.451])
        #ips = self_tensor_prod(ips, k)
        ops = [0.5, 0.5]
    
        pro = 1
        flag = 'fail'
        flag_count = 0
        result = catalytic_concentration2(ops, ips, cat_guess, k, dc)
        while flag == 'fail':
            result = catalytic_concentration2(ops, ips, cat_guess, k, dc)
            flag = result[4]
            flag_count += 1
            if flag_count == 10:
                flag = 'success'
        pro = result[1]
        #pro = result[2]
        cat_guess = result[3]
        
        cat_list.append(cat_guess)
        pro_cat.append(pro)
        dim_cat.append(dc)
    pro_nocat.append(pro_cat)  
    
"""

for i in range(nx):
    
    #ips = np.asarray([1.498-x[i], x[i]-0.498])
    ips = np.asarray([x[i], 1-x[i]])
    #ips = np.asarray([1.451-x[i], x[i]-0.451])
    #ips = self_tensor_prod(ips, k)
    ops = [0.5, 0.5]

    pro = 1
    flag = 'fail'
    flag_count = 0
    result = catalytic_concentration2(ops, ips, cat_guess, k, dc)
    while flag == 'fail':
        result = catalytic_concentration2(ops, ips, cat_guess, k, dc)
        flag = result[4]
        flag_count += 1
        if flag_count == 10:
            flag = 'success'
    #pro = result[1]
    pro = result[2]
    cat_guess = result[3]
    
    cat_list.append(cat_guess)
    pro_cat.append(pro)
    

k = 3
pro = 1
for i in range(nx):
    #ips = np.asarray([1.498-x[i], x[i]-0.498])
    #ips = np.asarray([1.451-x[i], x[i]-0.451])
    ips = np.asarray([x[i], 1-x[i]])
    ips = self_tensor_prod(ips, k)
    ops = [0.5, 0.5]
    ops = concat_zeros(ops, ips)
    pro = prob_of_transformation(ops, ips)
    
    pro_nocat.append(pro)

i = 0
flag = 1
while i < (len(pro_cat)) and flag == 1:
    i += 1
    #print(i)
    if pro_cat[i] - pro_nocat[i] > 1e-4:
        porbabbab = pro_cat[i]
        point = x[i]
        flag = 0

i=0
flag = 1
while i < (len(pro_cat)) and flag == 1:
    i += 1
    #print(i)
    if pro_nocat[i] - pro_cat[i] > 1e-5:
        porbabbab1 = pro_cat[i]
        point1 = x[i]
        flag = 0
print(point, point1)     


x1 = np.linspace(x_lim, 0.9999, nx)
plt.figure()
plt.plot(x1, pro_cat, x1, pro_nocat, point, porbabbab, 'o', point1, porbabbab1, 'o')
#plt.title('Max probability of transformation')
plt.ylabel('Probability of transformation')
plt.xlabel(r'Initial state $\alpha$')
plt.legend(["Catalytic", "Non-Catalytic"])
plt.grid()   
plt.savefig(dir_name+"/catalytic_gain"+".png", dpi=1000, format="png")
"""

x1 = np.linspace(x_lim, 0.995, nx)
plt.figure()
#plt.plot(x1, pro_nocat[0], x1, pro_nocat[1], x1, pro_nocat[2])#, x1, pro_nocat, point, porbabbab, 'o', point1, porbabbab1, 'o')
plt.plot(dim_cat, pro_nocat[0])#, dim_cat, pro_nocat[1])
#plt.title('Gain ')
plt.ylabel(r'Catalytic gain ($\eta$)')
#plt.yscale("log")
#plt.xlabel(r'Initial state $\alpha$')
plt.xlabel(r'Dimensions of catalyst')
#plt.legend(["k=2", "k=4","k=8"])
plt.legend([r"$\alpha$=0.75", r"$\alpha$=0.85"])
plt.grid()
#plt.savefig(dir_name+"/gain_varying_cat_dim"+".png", dpi=1000, format="png")

"""

