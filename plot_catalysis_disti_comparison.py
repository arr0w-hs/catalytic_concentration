#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 09:48:32 2024

@author: hsharma4
"""

import sys
import os
import pickle
import matplotlib.pyplot as plt

plt.rcParams.update({'font.size': 12})
sys.path.append(os.path.dirname(__file__))
dir_name = os.path.dirname(__file__)

#data_location = '/2024-06-14_cat_disti_comparison/94732'
data_location = '/2024-11-14_depol_channel_comparison/193222'#122431, 121820 #141740
with open(dir_name+'/data'+data_location+'.pkl', 'rb') as f:
    data_dict_loaded = pickle.load(f)
    f.close()

#print(data_dict_loaded.keys())

#fid_raw_list = data_dict_loaded["fid_raw_list"]
#prob_in_state_list = data_dict_loaded["alpha_list"]
prob_in_state_list = data_dict_loaded["prob_in_state_list"]

cat_fid = data_dict_loaded['cat_fid']

xmin = 0
xmax = 500
prob_in_state_list = prob_in_state_list[xmin:xmax]



#for i, ele in enumerate((prob_in_state_list)):
#    prob_in_state_list[i] = 1-prob_in_state_list[i]
    #print(ele)

#print(prob_in_state_list)
#x1 = np.linspace(np.min(fid_raw_list), 0.95, 100)



plt.figure()
plt.grid()
plt.plot(prob_in_state_list, data_dict_loaded["prob_cat_list"][xmin:xmax])#, s = 5)#, c = "red")#, "ob", alpha = 0.3)
plt.plot(prob_in_state_list, data_dict_loaded["prob_nocat_list"][xmin:xmax])#, s = 5)#, c = "blue")
plt.plot(prob_in_state_list, data_dict_loaded["prob_dist_list"][xmin:xmax])#, s = 5)#, c = "limegreen")
plt.plot(prob_in_state_list, data_dict_loaded["prob_cat_reuse_list"][xmin:xmax])#, s = 5)#, c = "orange")
#plt.gca().invert_xaxis()
#plt.scatter(fid_raw_list, dames_prob_list, s = 4, c = "orange")
#plt.title('Probability')
#plt.yscale("log")
#plt.xscale("log")
plt.ylabel('Probability of success')
#plt.xlabel('Raw state infidelity')
#plt.xlabel('Coherent error')
plt.xlabel('Probability of depolarisation')
plt.xticks(rotation=45)
plt.legend(["CEC", "NEC", "Distillation", "Catalyst reuse"])
#plt.savefig(dir_name +'/plots' + data_location + "_probability" + ".png", dpi=1000, format="png", bbox_inches = 'tight')
#plt.savefig(dir_name +'/plots' + data_location + "_probability" + ".pdf", dpi=1000, format="pdf", bbox_inches = 'tight')


fid_nocat_list = data_dict_loaded["fid_nocat_list"][xmin:xmax]
fid_cat_list = data_dict_loaded["fid_cat_list"][xmin:xmax]
fid_dist_list = data_dict_loaded["fid_dist_list"][xmin:xmax]
fid_cat_reuse_list = data_dict_loaded["fid_cat_reuse_list"][xmin:xmax]


plt.figure()
plt.grid()
plt.plot(prob_in_state_list, fid_cat_list)#, s = 5)#, c = "red")
plt.plot(prob_in_state_list, fid_nocat_list)#, s = 5)#, c = "blue")
plt.plot(prob_in_state_list, fid_dist_list)#, s = 5)#, c = "limegreen")
plt.plot(prob_in_state_list, fid_cat_reuse_list)#, s = 5)#, c = "orange")
#plt.gca().invert_xaxis()
#plt.scatter(fid_raw_list, fid_dames_list, s = 4, c = "orange")

#plt.scatter(kap, fip_nocat, s = 0.95)
#plt.title('Fidelity')
#plt.yscale("log")
plt.ylabel('Infidelity of final state')
#plt.xlabel('Raw state infidelity')
#plt.xlabel('Coherent error')
plt.xlabel('Probability of depolarisation')
plt.xticks(rotation=45)
plt.legend(["CEC", "NEC", "Distillation", "Catalyst reuse"])
#plt.savefig(dir_name +'/plots' + data_location + "_fidelity" + ".png", dpi=1000, format="png", bbox_inches = 'tight')
#plt.savefig(dir_name +'/plots' + data_location + "_fidelity" + ".pdf", dpi=1000, format="pdf", bbox_inches = 'tight')
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


plt.figure()
plt.grid()
plt.plot(prob_in_state_list, cat_fid)
#plt.gca().invert_xaxis()
#plt.scatter(fid_raw_list, fid_dames_list, s = 4, c = "orange")

#plt.scatter(kap, fip_nocat, s = 0.95)
#plt.title('Fidelity')
#plt.yscale("log")
plt.ylabel('Fidelity')
#plt.xlabel('Coherent error')
plt.xlabel('Probability of depolarisation')
plt.xticks(rotation=45)
#plt.legend(["CEC", "NEC", "Distillation", "Catalyst reuse"])
plt.savefig(dir_name +'/plots' + data_location + "_cat_fidelity" + ".png", dpi=1000, format="png", bbox_inches = 'tight')
#plt.savefig(dir_name +'/plots' + data_location + "_fidelity" + ".pdf", dpi=1000, format="pdf", bbox_inches = 'tight')
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
print()
