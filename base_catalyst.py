#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 13:36:20 2023

@author: hsharma4
"""

import numpy as np
import scipy as sc
#mport qutip as qt


def self_tensor_prod(vec, num_of_copies):
    """function for tensoring the state with itself num_of_copies times"""
    dim = np.shape(vec)[0]
    prod_vec = vec
    for i in range(num_of_copies-1):
        prod_vec = np.reshape(np.tensordot(prod_vec, vec, 0), dim**(i+2))
    prod_vec = np.sort(prod_vec)[::-1]
    return prod_vec


def ent_mono(vec, m_val):
    """function for calculating entanglement monotone"""
    ent_monotone = 0
    for i in range(m_val, np.shape(vec)[0]):
        ent_monotone += vec[i]
    return ent_monotone


def prob_of_transformation2(phi, psi):
    """ function for calculating the probability of transformation
        phi is output state, psi is input state"""

    assert np.shape(psi)[0] >= np.shape(phi)[0]

    p_list = np.zeros(np.shape(psi)[0])
    for i in range(np.shape(psi)[0]):
        if ent_mono(phi, i) != 0:
            p_list[i] = ent_mono(psi, i)/ent_mono(phi, i)
        else:
            p_list[i] = 1000
            #print("one of the ent monotones is zero")
    return np.min(p_list)


def func_prob(cat_1, in_state, out_state):
    """function to find the negative of probability with the use of catalyst"""

    if np.linalg.norm(cat_1, ord=1) != 0:
        cat_1 = cat_1/np.linalg.norm(cat_1, ord=1)

    in_1 = np.reshape(np.tensordot(in_state, cat_1, 0), np.shape(in_state)[0]*np.shape(cat_1)[0])
    out_1 = np.reshape(np.tensordot(out_state, cat_1, 0), np.shape(out_state)[0]*np.shape(cat_1)[0])
    in_1 = np.sort(in_1)[::-1]
    out_1 = np.sort(out_1)[::-1]

    if prob_of_transformation2(out_state, in_state) !=0:
        func = -1*prob_of_transformation2(out_1, in_1)
    else:
        func = 0

    return func


def catalytic_concentration(outstate, instate, num_k, d_c):
    """instate is input, outstate is output state,
        num_k is the number of input states to be tensored
        d_c is the dimension of the catalyst"""

    instate = instate/np.sum(instate)#, ord=1)
    instate_tensored = self_tensor_prod(instate, num_k)

    bnds = []
    for _ in range(d_c):
        bnds.append((0, 1))
    instate0 = np.sqrt(instate[0])
    cat_guess = [instate0, 1-instate0]#np.random.randint(1, 100000, size=d_c)
    cat_guess = cat_guess/np.sum(cat_guess)

    res = sc.optimize.minimize(func_prob, cat_guess, args = (instate_tensored, outstate),
                                method='SLSQP', bounds = bnds)


    cat_final = res.x

    if np.sum(cat_final) == 0:
        flg = "fail"
        print("fail encountered")
    else:
        cat_final = cat_final/np.sum(cat_final)
        flg = 'success'
    #cat_final = np.sort(cat_final)[::-1]

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



# def catalytic_concentration2(outstate, instate, catguess, num_k, d_c):
#     instate = instate/np.linalg.norm(instate, ord=1)
#     instate_tensored = self_tensor_prod(instate, num_k)

#     bnds = []
#     for ii in range(d_c):
#         bnds.append((0, 1))
#     #cat_guess = np.random.randint(1, 100000, size=dc)
#     #cat_guess = cat_guess/np.linalg.norm(cat_guess, ord=1)

#     res = sc.optimize.minimize(func_prob, catguess, args = (instate_tensored, outstate),
#                                method='SLSQP', bounds = bnds)


#     cat_final = res.x
#     if np.linalg.norm(cat_final, ord=1) == 0:
#         flg = "fail"
#         print("fail encountered")
#         pcr = 0
#     else:
#         cat_final = cat_final/np.linalg.norm(cat_final, ord=1)
#         flg = 'success'
#         pcr = -1*res.fun
#     cat_final = np.sort(cat_final)[::-1]


#     pncr = prob_of_transformation2(outstate, instate_tensored)
#     gain = pcr/pncr
#     return pncr, gain, pcr, cat_final, flg
