#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 13:36:20 2023

@author: hsharma4
"""

import numpy as np
import scipy as sc
from qutip import *
from qutip.measurement import measure, measurement_statistics, measure_observable

#func for tensor product of a state with itself
def self_tensor_prod(vec, num_of_copies):
    dim = np.shape(vec)[0]
    prod_vec = vec
    for i in range(num_of_copies-1):
        prod_vec = np.reshape(np.tensordot(prod_vec, vec, 0), dim**(i+2))
    prod_vec = np.sort(prod_vec)[::-1]
    return prod_vec

#function for calculating entanglement monotone
def ent_mono(vec, m):
    em = 0
    for i in range(m, np.shape(vec)[0]):
        em += vec[i]
    return em

#function for calculating the probability of transformation
#phi is output state, psi is input state
def prob_of_transformation2(phi, psi, *args):
    #psi = kwargs['initial_state']
    if np.shape(psi)[0] < np.shape(phi)[0]:
        raise Exception("Incoherent dimensions of states")

    p = np.zeros(np.shape(psi)[0])
    for i in range(np.shape(psi)[0]):
        if ent_mono(phi, i) != 0:
            p[i] = ent_mono(psi, i)/ent_mono(phi, i)
        else:
            p[i] = 1000
            #print("one of the ent monotones is zero")
    return np.min(p)

#function to find the negative of probability with the use of catalyst
def func_prob(cat_1, in_state, out_state):
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
        cat_final = [0.50, 0.50]

    if gain > 50:
        print("gain > 50")
        print(cat_final, "catalyst state")
        print(instate_tensored, "initial state tensored")
    return pncr, gain, pcr, cat_final, flg

def catalytic_concentration2(outstate, instate, catguess, num_k, d_c):
    instate = instate/np.linalg.norm(instate, ord=1)
    instate_tensored = self_tensor_prod(instate, num_k)

    bnds = []
    for ii in range(d_c):
        bnds.append((0, 1))
    #cat_guess = np.random.randint(1, 100000, size=dc)
    #cat_guess = cat_guess/np.linalg.norm(cat_guess, ord=1)

    res = sc.optimize.minimize(func_prob, catguess, args = (instate_tensored, outstate),
                               method='SLSQP', bounds = bnds)


    cat_final = res.x
    if np.linalg.norm(cat_final, ord=1) == 0:
        flg = "fail"
        print("fail encountered")
        pcr = 0
    else:
        cat_final = cat_final/np.linalg.norm(cat_final, ord=1)
        flg = 'success'
        pcr = -1*res.fun
    cat_final = np.sort(cat_final)[::-1]


    pncr = prob_of_transformation2(outstate, instate_tensored)
    gain = pcr/pncr
    return pncr, gain, pcr, cat_final, flg

'''
res_dif_ini_state2 = []
dc = 2
k = 2


for i in range(1):
    dc = 2**(i+1)

    cat_guess = np.random.randint(1, 100000, size=dc)
    cat_guess = cat_guess/np.linalg.norm(cat_guess, ord=1)

    res_ca = []
    for j in range(700, 1000):
        pro = 1
        flag = 'fail'
        flag_count = 0
        input_state = [j/1000, 1-j/1000]
        output_state = [0.5, 0.5]
        result = catalytic_concentration2(output_state, input_state, cat_guess, k, dc)
        while flag == 'fail':
            result = catalytic_concentration2(output_state, input_state, cat_guess, k, dc)
            flag = result[4]
            flag_count += 1
            if flag_count == 10:
                flag = 'success'
        pro = result[2]
        cat_guess = result[3]

        res_ca.append(pro)

    res_dif_ini_state2.append(res_ca)

a = np.linspace(0.7, 1, 300)

plt.plot(a, res_dif_ini_state2[0])'''