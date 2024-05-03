#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep  4 13:33:42 2023

@author: hsharma4
code for finding robustness of slocc for imperfect initial state and perfect catalyst
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
from base_locc import *
from base_slocc import *
from base_state_change import *
from base_catalyst import *
from base_analysis import *

fid = []
prob = []
nrmsd = []
doub_der = []
left_wid = []
right_wid = []
fid_cat = []
prob_cat = []
doub_der_cat = []
left_wid_cat = []
right_wid_cat = []
nrmsd_cat = []

pro_cat = []
pro_nocat = []

ops = [0.5, 0.5]

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
probab = []
cat_list = []
for i in range(nx):
    #ips = np.asarray([1.498-x[i], x[i]-0.498])
    ips = np.asarray([x[i], 1-x[i]])
    ##ips = self_tensor_prod(ips, k)
    ops = [0.5, 0.5]
    
    pro = 1
    pronc = 1
    flag = 'fail'
    flag_count = 0
    result = catalytic_concentration2(ops, ips, cat_guess, k, dc)
    while flag == 'fail':
        result = catalytic_concentration2(ops, ips, cat_guess, k, dc)
        flag = result[4]
        flag_count += 1
        if flag_count == 10:
            flag = 'success'
    pro = result[2]
    cat_guess = result[3]
    pronc = result[0]
    cat_list.append(cat_guess)
    pro_cat.append(pro)
    pro_nocat.append(pronc)
    ips_t = self_tensor_prod(ips, k)
    #print(ips_t)
    #print(cat_guess)
    probab.append(prob_of_transformation2(cat_guess, ips_t))
    
print(np.shape(cat_list))


"""
for i in range(nx):
    ips = np.asarray([x[i], 1-x[i]])
    ips = self_tensor_prod(ips, 2)
    #print(ips)
    #print(cat_list[i])
"""

x1 = np.linspace(x_lim, 0.9999, nx)
plt.plot(x1, probab)
"""
for i in range(nx):
    print(i)
    #ips = np.asarray([1.498-x[i], x[i]-0.498])
    ips = np.asarray([x[i], 1-x[i]])
    #ips = np.asarray([1.451-x[i], x[i]-0.451])
    ips = self_tensor_prod(ips, k)
    ops = [0.5, 0.5]
    fidy = []
    proby = []
    fidy_cat = []
    proby_cat = []
    
    cat_state = cat_list[i]
    ips_cat = np.reshape(np.tensordot(ips, cat_state, 0), np.shape(ips)[0]*np.shape(cat_state)[0])
    ips_cat = np.sort(ips_cat)[::-1]
    
    ops_cat = np.reshape(np.tensordot(ops, cat_state, 0), np.shape(ops)[0]*np.shape(cat_state)[0])
    ops_cat = np.sort(ops_cat)[::-1]
    
    
    for j in range(ny):
        error_ips = np.asarray([y[j], 1-y[j]])
        error_ips = self_tensor_prod_ver2(error_ips, k)
        #error_ips = np.sort(error_ips)[::-1]
        error_ips_cat = np.reshape(np.tensordot(error_ips, cat_state, 0), np.shape(error_ips)[0]*np.shape(cat_state)[0])
        error_ips_cat = np.sort(error_ips_cat)[::-1]
        
        out_dm_cat, fid_val_cat, prob_val_cat = state_transformation(ops_cat, ips_cat, error_ips_cat)
        fidy_cat.append(fid_val_cat)
        proby_cat.append(prob_val_cat)#*fid_val_cat)
        
        out_dm, fid_val, prob_val = state_transformation(ops, ips, error_ips)
        fidy.append(fid_val)
        proby.append(prob_val)#*fid_val)
    
    
    fid.append(fidy)
    fid_cat.append(fidy_cat)
    prob.append(proby)
    prob_cat.append(proby_cat)
    
    nrmsd_cat.append(fwhm_calculator(fidy_cat)/(4*x[i])*100) 
    doub_der_cat.append(-1/double_derivative(fidy_cat, y))
    left_wid_cat.append(left_width(fidy_cat)/ny/(x[i])*100)
    right_wid_cat.append(right_width(fidy_cat)/ny/(x[i])*100)

    nrmsd.append(fwhm_calculator(fidy)/(4*x[i])*100)
    doub_der.append(-1/double_derivative(fidy, y))
    left_wid.append(left_width(fidy)/ny/(x[i])*100)
    right_wid.append(right_width(fidy)/ny/(x[i])*100)


fid = np.reshape(fid, [nx,ny]) 
fid = np.flip(fid, (0))
fid_cat = np.reshape(fid_cat, [nx,ny]) 
fid_cat = np.flip(fid_cat, (0))
prob = np.reshape(prob, [nx,ny]) 
prob = np.flip(prob, (0))
prob_cat = np.reshape(prob_cat, [nx,ny]) 
prob_cat = np.flip(prob_cat, (0))
"""

x1 = np.linspace(x_lim, 0.9999, nx)
plt.figure()
plt.plot(x1, probab, x1, pro_nocat)
#plt.title('Probability comparison')
plt.ylabel('Probability')
plt.legend(["Catalytic probability", "Non-catalytic probability"])
plt.xlabel(r'Initial state $\alpha$')
plt.grid()
#plt.savefig(dir_name+"/catalytic_prob_k2"+".png", dpi=1000, format="png")

"""
plt.figure()
plt.imshow(fid, aspect = 'auto', interpolation='nearest', 
           extent = ( y_lim, 1, x_lim, 1), vmin=None, vmax=None)#,cmap='hot'
plt.colorbar() 
plt.title("Fidelity of SLOCC transformation")
plt.ylabel(r'Ideal state $\alpha$')
plt.xlabel(r"Initial state $\alpha$' ")
#plt.savefig(dir_name+"/fidelity"+".png", dpi=1000, format="png")


plt.figure()
plt.imshow(fid_cat, aspect = 'auto', interpolation='nearest', 
           extent = (y_lim, 1, x_lim, 1), vmin=None, vmax=None)
plt.colorbar() 
plt.title("Fidelity of Catalytic SLOCC transformation")
plt.ylabel(r'Ideal state $\alpha$')
plt.xlabel(r"Initial state $\alpha$' ")
#plt.savefig(dir_name+"/fidelity_catalytic"+".png", dpi=1000, format="png")


plt.figure()
plt.imshow(prob, aspect = 'auto', interpolation='nearest', 
           extent = (y_lim, 1, x_lim, 1), vmin=None, vmax=None)
plt.colorbar() 
plt.title("Probability of SLOCC transformation")
plt.ylabel(r'Ideal state $\alpha$')
plt.xlabel(r"Initial state $\alpha$' ")
#plt.savefig(dir_name+"/prob"+".png", dpi=1000, format="png")


plt.figure()
plt.imshow(prob_cat, aspect = 'auto', interpolation='nearest', 
           extent = (y_lim, 1, x_lim, 1), vmin=None, vmax=None)
plt.colorbar() 
plt.title("Probability of Catalytic SLOCC transformation")
plt.ylabel(r'Ideal state $\alpha$')
plt.xlabel(r"Initial state $\alpha$' ")
#plt.savefig(dir_name+"/prob_catalytic"+".png", dpi=1000, format="png")


plt.figure()
r_prob = prob_cat/prob
#r_prob = np.log(r_prob)
plt.imshow(r_prob, aspect = 'auto', interpolation='nearest', 
           extent = (y_lim, 1, x_lim, 1), vmin=None, vmax=None)
plt.colorbar() 
plt.title("Ratio of probability of Catalytic SLOCC transformation")
plt.ylabel(r'Ideal state $\alpha$')
plt.xlabel(r"Initial state $\alpha$' ")
#plt.savefig(dir_name+"/ratio_prob"+".png", dpi=1000, format="png")


plt.figure()
ratio = fid_cat/fid
#ratio = np.log(ratio)
plt.imshow(ratio, aspect = 'auto', interpolation='nearest', 
           extent = (y_lim, 1, x_lim, 1), vmin=None, vmax=None)
plt.colorbar() 
plt.title("Ratio of fidelities of SLOCC transformation")
plt.ylabel(r'Ideal state $\alpha$')
plt.xlabel(r"Initial state $\alpha$' ")
#plt.savefig(dir_name+"/ratio_fidelity"+".png", dpi=1000, format="png")

plt.figure()
#x = np.linspace(x_lim, 0.999, nx)
#x1 = np.linspace(x_lim, 0.999, nx)
plt.plot(x, doub_der, x, doub_der_cat)
#plt.yscale('log')
plt.title('Curvature of fidelity plot')
plt.legend(["No catalyst", "catalyst"])
plt.ylabel('Curvature')
plt.xlabel(r'Ideal state $\alpha$')
#plt.savefig(dir_name+"/curvature"+".png", dpi=1000, format="png")


plt.figure()    
plt.plot(x, nrmsd, x, nrmsd_cat)
plt.title('Error for >0.95 fidelity')
plt.ylabel('Allowed error in Initial state (%)')
plt.xlabel(r'Ideal state $\alpha$')
plt.legend(["No catalyst", "catalyst"])
#plt.savefig(dir_name+"/FW0.9M"+".png", dpi=1000, format="png")


plt.figure()
plt.plot(x, left_wid, x, left_wid_cat)
#plt.yscale('log')
plt.title('Width at 0.9*maximum')
plt.ylabel('Width on the left')
plt.xlabel(r'Ideal state $\alpha$')
plt.legend(["No catalyst", "catalyst"])
#plt.savefig(dir_name+"/left_width"+".png", dpi=1000, format="png")


plt.figure()
plt.plot(x, right_wid, x, right_wid_cat)
#plt.yscale('log')
plt.title('Width at 0.9*maximum')
plt.ylabel('Width on the right')
plt.xlabel(r'Ideal state $\alpha$')
plt.legend(["No catalyst", "catalyst"])
#plt.savefig(dir_name+"/right_width"+".png", dpi=1000, format="png")
"""
