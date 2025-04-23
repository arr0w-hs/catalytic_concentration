#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 14:23:38 2025

@author: hsharma4

function for calling and directly plotting the depol channel comparison
"""
import sys
import os

import pickle
import matplotlib.pyplot as plt
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

import qutip as qt
from pathlib import Path
from base_transform import catalytic_conversion, non_catalytic_conversion, catalytic_conversion_reuse, closest_pure_state
from base_distillation import distillation
from base_depol_channels import  new_state_pauli_x1
import numpy as np
import pandas as pd
plt.rcParams.update({'font.size': 12})
from plot_perf import plot_perf

def performance_comparison(param_list, coherent_err = True):
    """
        for comparing the performance of our protocols in presence of coherent 
        and depolarising errors

        Parameters
        ----------
        a_max: the maximum value of the coherent error
        a_min: the minimum value of coherent error (default is set to zero)
        p_const: the value of depol error is fixed to this value while 
                 value of coherent error is changed

        p_max: maximum value of depol error
        p_min: minimum value of depol error
        a_const: the constant value of coherent error while the value of 
                 depol error is varied
                 
        num_points: is the number of points in the plot

        Returns
        -------
        data_dict: a dictionary with the values of fidelity and probabilities
                   for each protocol.
                   the value of fidelity of the catalyst state before and after
                   it is used
        """


    alpha_list = []
    prob_in_state_list = []
    
    cat_fidelity = []

    cat_state = []
    cat_fid_reuse = []

    fid_raw_list = []
    fid_raw_one = []
    fid_raw_two = []
    fid_nocat_list = []
    fid_dist_list = []
    fid_cat_list = []
    fid_cat_reuse_list = []
    
    prob_nocat_list = []
    prob_dist_list = []
    prob_cat_list = []
    prob_cat_reuse_list = []

    cat_fid = []
    eigen_val_list = []


    if coherent_err:
        error_list = np.linspace(1-param_list[5], 1-param_list[0], num=param_list[4])
    else:
        error_list = np.linspace(1-param_list[6], 1-param_list[1], num=param_list[4])



    for i in error_list:

        if coherent_err:
            alpha = i
            prob_in_state = param_list[3]
        else:
            prob_in_state = i
            alpha = param_list[2]

        #prob_in_state = 0.95
        #alpha = 0.9999 - i/num_points*0.20
        # print(alpha)

        final_state = new_state_pauli_x1(alpha, prob_in_state)
    
        ideal_state = new_state_pauli_x1(1, 1)
    
        _, eigen_val = closest_pure_state(final_state)
        eigen_val_list.append(eigen_val)
    
    
        fid_cat, prob_cat, post_locc_state, carbon_cat_st = catalytic_conversion(final_state)
        fid_nocat, prob_nocat, post_locc_state_nocat, _ = non_catalytic_conversion(final_state)
        fid_dist, prob_dist, _ = distillation(final_state, 0)
    
        cat_post = post_locc_state.ptrace([2,5])
    
        fid_reuse, prob_reuse, out_state_reuse, flag = catalytic_conversion_reuse(final_state, cat_post)
        cat_post_reuse = out_state_reuse.ptrace([2,5])
    
        cat_fid.append(qt.fidelity(carbon_cat_st, cat_post))
        cat_fid_reuse.append(qt.fidelity(cat_post_reuse, carbon_cat_st))
    
    
    
        fid_cat_list.append(1-fid_cat)
        fid_nocat_list.append(1-fid_nocat)
        fid_dist_list.append(1-fid_dist)
        fid_cat_reuse_list.append(1-fid_reuse)
    
        prob_cat_list.append(prob_cat)
        prob_nocat_list.append(prob_nocat)
        prob_dist_list.append(prob_dist)
        prob_cat_reuse_list.append(prob_reuse)
    
        cat_state.append((carbon_cat_st[0]))
        cat_fid_post = qt.fidelity(carbon_cat_st, cat_post)
        cat_fidelity.append(cat_fid_post)
    
    
        bell_st = 1/np.sqrt(2)*(qt.tensor(qt.basis(2,0), qt.basis(2,0)) +
                                qt.tensor(qt.basis(2,1), qt.basis(2,1)))
        raw_fid = np.sqrt(qt.fidelity(ideal_state, final_state))
        fid_raw_one.append(qt.fidelity(bell_st, final_state.ptrace([1,3])))
        fid_raw_two.append(qt.fidelity(bell_st, final_state.ptrace([2,4])))
        fid_raw_list.append(1-(raw_fid))
    
        prob_in_state_list.append(1-prob_in_state)
        alpha_list.append(1-alpha)
    
    data_dict = {
        "alpha_list": alpha_list,
        "prob_in_state_list": prob_in_state_list,
        "fid_cat_list": fid_cat_list,
        "fid_nocat_list": fid_nocat_list,
        "fid_dist_list": fid_dist_list,
        "fid_cat_reuse_list": fid_cat_reuse_list,
        "prob_cat_list": prob_cat_list,
        "prob_nocat_list": prob_nocat_list,
        "prob_dist_list": prob_dist_list,
        "prob_cat_reuse_list": prob_cat_reuse_list,
        "cat_fid_reuse": cat_fid_reuse,
        "cat_fid": cat_fid,
        }

    return data_dict


def perf_comparison(a_max, p_max, a_const, p_const, num_points = 20,
                    a_min = 0, p_min = 0, save_data=True, save_plot=False):
    """
        for comparing the performance of our protocols in presence of coherent 
        and depolarising errors. this code creates two folders: 
        (1) called "data" for saving data and (2) "plots" for saving plots.
        Then subfolders created with the date string as the name. Each 
        subfolder has the plots and data saved with time string as the name.

        Parameters
        ----------
        a_max: the maximum value of the coherent error
        a_min: the minimum value of coherent error (default is set to zero)
        p_const: the value of depol error is fixed to this value while 
                 value of coherent error is changed

        p_max: maximum value of depol error
        p_min: minimum value of depol error set to 0 by default
        a_const: the constant value of coherent error while the value of 
                 depol error is varied
                 
        num_points: is the number of points in the plot
        save_data: saves the data "data_dict" in a form of a pickle file unless
                   set to False

        Returns
        -------
        data_dict: a dictionary of dictionaries with two dictionaries
                   coherent and depol with each dictionary having
                   the values of fidelity
                   and probabilities for each protocol.
                   the value of fidelity of the catalyst state before and after
                   it is used
        time_str: is the time string that corresponds to a run
    """

    ts = pd.Timestamp.today(tz = 'Europe/Stockholm')
    date_str = str(ts.date())
    time_str = ts.time()
    time_str = str(time_str.hour)+ str(time_str.minute) + str(time_str.second)

    print("time string = ", time_str)
    data_folder = Path(os.path.join(dir_name,"data"))
    plot_folder = Path(os.path.join(dir_name,"plots"))
    if not (data_folder).exists() and save_data:
        os.mkdir(data_folder)

    data_directory = os.path.join(data_folder, date_str)
    plots_directory = os.path.join(plot_folder, date_str)

    date_folder = Path(data_directory)

    if not date_folder.exists():
        os.mkdir(data_directory)

    param_list = [a_max, p_max, a_const, p_const, num_points, a_min, p_min]
    data_dict = {}
    data_dict["coherent"] = performance_comparison(param_list, coherent_err = True)
    data_dict["depol"] = performance_comparison(param_list, coherent_err = False)

    with open(data_directory+"/" +time_str +'.pkl', 'wb') as f:
        pickle.dump(data_dict, f)

    plot_perf(time_str, date_str, save_plot)

    return data_dict, time_str

if __name__=="__main__":

    perf_comparison(0.19, 0.19, 0.9, 0.95, num_points = 19, a_min = 0.0001, p_min = 0.0001, save_plot=True)
