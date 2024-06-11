#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:48:32 2024

@author: hsharma4
"""

import sys
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

data_location = '/2024-05-29_cat_disti_comparison/151057'
#data_location = '/2024-04-29_depol_channel_comparison/16531'
with open(dir_name+'/data'+data_location+'.pkl', 'rb') as f:
    data_dict_loaded = pickle.load(f)
    f.close()

print(data_dict_loaded.keys())

fid_raw_list = data_dict_loaded["fid_raw_list"]
#alpha_list = data_dict_loaded["alpha_list"]
#prob_in_state_list = data_dict_loaded["prob_in_state_list"]
prob_in_state_list = data_dict_loaded["fid_raw_list"]

x1 = np.linspace(np.min(fid_raw_list), 0.95, 100)

plt.figure()
plt.grid()
plt.scatter(prob_in_state_list, data_dict_loaded["prob_cat_list"], s = 5, c = "red")#, "ob", alpha = 0.3)
plt.scatter(prob_in_state_list, data_dict_loaded["prob_nocat_list"], s = 5, c = "blue")
plt.scatter(prob_in_state_list, data_dict_loaded["prob_dist_list"], s = 5, c = "limegreen")
plt.scatter(prob_in_state_list, data_dict_loaded["prob_cat_reuse_list"], s = 5, c = "orange")
#plt.gca().invert_xaxis()
#plt.scatter(fid_raw_list, dames_prob_list, s = 4, c = "orange")
#plt.title('Probability')
#plt.yscale("log")
#plt.xscale("log")
plt.ylabel('Probability of success')
#plt.xlabel('Raw state infidelity')
#plt.xlabel('Error in state ("coherent error")')
plt.xlabel('Error probability ("mixed-ness")')
plt.xticks(rotation=45)
plt.legend(["Catalytic", "Non-catalytic", "Distillation", "Catalyst reuse"])
#plt.savefig(dir_name +'/plots' + data_location + "_probability" + ".png", dpi=800, format="png", bbox_inches = 'tight')
"""
z = np.polyfit(raw_fid_list, cat_prob, 2)
p = np.poly1d(z)
plt.plot(x1,p(x1),"--", c = "red", linewidth = 0.7)

z = np.polyfit(raw_fid_list, nocat_prob, 2)
p = np.poly1d(z)
plt.plot(x1,p(x1),"--", c = "blue", linewidth = 0.7)

z = np.polyfit(raw_fid_list, dist_prob_list, 2)
p = np.poly1d(z)
plt.plot(x1,p(x1),"--", c = "limegreen", linewidth = 0.7)

z = np.polyfit(raw_fid_list, dames_prob_list, 2)
p = np.poly1d(z)
plt.plot(x1,p(x1),"--", c = "orange", linewidth = 0.7)


#plt.plot(x1, x1, linewidth = 0.5)
z = np.polyfit(raw_fid_list, fid_cat, 2)
p = np.poly1d(z)
plt.plot(x1,p(x1),"--", c = "red", linewidth = 0.7)

z = np.polyfit(raw_fid_list, fid_no_cat, 2)
p = np.poly1d(z)
plt.plot(x1,p(x1),"--", c = "blue", linewidth = 0.7)

z = np.polyfit(raw_fid_list, fid_dist_list, 2)
p = np.poly1d(z)
plt.plot(x1,p(x1),"--", c = "limegreen", linewidth = 0.7)

z = np.polyfit(raw_fid_list, fid_dames_list, 2)
p = np.poly1d(z)
plt.plot(x1,p(x1),"--", c = "orange", linewidth = 0.7)

x1 = np.linspace(np.min(raw_fid_list), np.max(raw_fid_list), 100)
"""

plt.figure()
plt.grid()
plt.scatter(prob_in_state_list, data_dict_loaded["fid_cat_list"], s = 5, c = "red")
plt.scatter(prob_in_state_list, data_dict_loaded["fid_nocat_list"], s = 5, c = "blue")
plt.scatter(prob_in_state_list, data_dict_loaded["fid_dist_list"], s = 5, c = "limegreen")
plt.scatter(prob_in_state_list, data_dict_loaded["fid_cat_reuse_list"], s = 5, c = "orange")
#plt.gca().invert_xaxis()
#plt.scatter(fid_raw_list, fid_dames_list, s = 4, c = "orange")

#plt.scatter(kap, fip_nocat, s = 0.95)
#plt.title('Fidelity')
#plt.yscale("log")
plt.ylabel('Infidelity of final state')
#plt.xlabel('Raw state infidelity')
#plt.xlabel('Error in state ("coherent error")')
plt.xlabel('Error probability ("mixed-ness")')
plt.xticks(rotation=45)
plt.legend(["Catalytic", "Non-catalytic", "Distillation", "Catalyst reuse"])
#plt.savefig(dir_name +'/plots' + data_location + "_fidelity" + ".png", dpi=1000, format="png", bbox_inches = 'tight')
"""
plt.figure()
plt.grid()
plt.scatter(dist_list, fid_raw_list, s = 5, c = "blue")
plt.yscale("log")
plt.ylabel('Raw pair fidelity')
#plt.xlabel('Raw state fidelity')
plt.xlabel('Distance (km)')
plt.xticks(rotation=45)
#plt.legend(["Catalytic", "Non-catalytic", "Distillation", "DEJMPS"])
#plt.savefig(dir_name+"/fidelity_vs_dist"+".png", dpi=1000, format="png")
"""
