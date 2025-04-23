#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 15:33:23 2025

@author: hsharma4
for plottin the data obtained
"""

import sys
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams.update({'font.size': 12})
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

def plot_perf(time_str, date_str, saving):
    
    data_folder = Path(os.path.join(dir_name,"data"))
    plot_folder = Path(os.path.join(dir_name,"plots"))
    if not (plot_folder).exists():
        os.mkdir(plot_folder)


    plots_directory = os.path.join(plot_folder, date_str)
    data_directory = os.path.join(data_folder, date_str)
    plot_folder = Path(plots_directory)
    if not plot_folder.exists() and saving:
        os.mkdir(plots_directory)

    with open(data_directory +"/" + time_str +'.pkl', 'rb') as f:
        data_dict_dict = pickle.load(f)
        f.close()


    for i in range(2):
        if i == 0:
            data_dict_loaded = data_dict_dict["coherent"]
            prob_in_state_list = data_dict_loaded["alpha_list"]
            error_name = "Coherent error"
        else:
            data_dict_loaded = data_dict_dict["depol"]
            prob_in_state_list = data_dict_loaded["prob_in_state_list"]
            error_name = "Probability of depolarisation"

        cat_fid = data_dict_loaded['cat_fid']
        cat_fid_reuse = data_dict_loaded['cat_fid_reuse']
        
        xmin = 0
        xmax = 500
        prob_in_state_list = prob_in_state_list[xmin:xmax]
        fs = 15

        
        plt.figure()
        plt.grid()
        plt.plot(prob_in_state_list, data_dict_loaded["prob_cat_list"][xmin:xmax],'o-', label = 'CEC')
        plt.plot(prob_in_state_list, data_dict_loaded["prob_nocat_list"][xmin:xmax],'.-', label = 'NEC')
        plt.plot(prob_in_state_list, data_dict_loaded["prob_dist_list"][xmin:xmax], 'v-', label = 'Distillation')
        plt.plot(prob_in_state_list, data_dict_loaded["prob_cat_reuse_list"][xmin:xmax], 'x-', label = 'Catalyst reuse')
        
        plt.ylabel('Probability of success', fontsize=fs)
        plt.xlabel(error_name, fontsize=fs)
        plt.xticks(rotation=45, fontsize=fs)
        plt.yticks(fontsize = fs)
        plt.legend(fontsize = fs,
                   handlelength=1.3, handleheight=0.5, labelspacing = 0.15)
        if saving:
            # plt.savefig(str(plot_folder)+ "/"+time_str+"_probability" + ".svg", dpi=1000, format="svg", bbox_inches = 'tight')
            plt.savefig(str(plot_folder) + "/"+time_str+"_probability_" +error_name+ ".pdf", dpi=1000, format="pdf", bbox_inches = 'tight')

        
        fid_nocat_list = data_dict_loaded["fid_nocat_list"][xmin:xmax]
        fid_cat_list = data_dict_loaded["fid_cat_list"][xmin:xmax]
        fid_dist_list = data_dict_loaded["fid_dist_list"][xmin:xmax]
        fid_cat_reuse_list = data_dict_loaded["fid_cat_reuse_list"][xmin:xmax]
        plt.figure()
        plt.grid()
        plt.plot(prob_in_state_list, fid_cat_list, 'o-', label = 'CEC')
        plt.plot(prob_in_state_list, fid_nocat_list, '.-', label = 'NEC')
        plt.plot(prob_in_state_list, fid_dist_list, 'v-', label = 'Distillation')
        plt.plot(prob_in_state_list, fid_cat_reuse_list,'x-', label = 'Catalyst reuse')
        
        plt.ylabel('Infidelity of final state', fontsize=fs)
        plt.xlabel(error_name, fontsize=fs)
        plt.xticks(rotation=45, fontsize=fs)
        plt.yticks(fontsize = fs)
        plt.legend(fontsize = fs,
                   handlelength=1.3, handleheight=0.5, labelspacing = 0.15)
        if saving:
            # plt.savefig(str(plot_folder)+ "/"+time_str+"_fidelity" + ".svg", dpi=1000, format="svg", bbox_inches = 'tight')
            plt.savefig(str(plot_folder) + "/"+time_str+"_fidelity_" +error_name + ".pdf", dpi=1000, format="pdf", bbox_inches = 'tight')
        
        
        
        cat_fid = [1-ele for ele in cat_fid]
        cat_fid_reuse = [1-ele for ele in cat_fid_reuse]

        plt.figure()
        plt.grid()
        plt.plot(prob_in_state_list, cat_fid, 'o-', label = 'Catalyst')
        plt.plot(prob_in_state_list, cat_fid_reuse, 'x-', label = 'Reuse', color="tab:red")
        plt.ylabel('Catalyst infidelity', fontsize=fs)
        plt.xlabel(error_name, fontsize=fs)
        plt.xticks(rotation=45, fontsize=fs)
        if i==0:
            plt.yticks(np.arange(0, 0.101, step=0.02))

        plt.yticks(fontsize=fs)
        plt.legend(fontsize = fs,
                   handlelength=1.3, handleheight=0.5, labelspacing = 0.15)
        if saving:
            # plt.savefig(str(plot_folder) + "/"+time_str+ "_cat_fidelity" + ".svg", dpi=1000, format="svg", bbox_inches = 'tight')
            plt.savefig(str(plot_folder) + "/"+time_str+ "_cat_fidelity_" +error_name+ ".pdf", dpi=1000, format="pdf", bbox_inches = 'tight')
        plt.show()


if __name__=="__main__":
    plot_perf("131522", "2025-04-23", saving=True)
