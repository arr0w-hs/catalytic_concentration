
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:42:23 2023

@author: hsharma4
"""

import numpy as np
import scipy as sc
import math as math
import pandas as pd
import matplotlib.pyplot as plt
from qutip import *
from qutip.measurement import measure, measurement_statistics, measure_observable

"""basic operators"""
X = sigmax()
Z = sigmaz()
I = qeye(2)
H = 1/math.sqrt(2)*(X+Z)
ketbra0 = ket2dm(basis(2, 0))
ketbra1 = ket2dm(basis(2, 1))
plus = ket2dm((basis(2, 0)+basis(2, 1)).unit())
minus = ket2dm((basis(2, 0)-basis(2, 1)).unit())

"""psn is of the form |ph>|sp_a>|nu_a>|sp_b>|nu_b>
and psn_dm is a density matrix with all the information"""



"""
******************************************************
******************************************************
Functions for measurements
******************************************************
******************************************************
"""

"""measurement of photon in X basis"""
def measure_photon_x(psn):
    pz0 = tensor(plus, I, I, I, I)
    pz1 = tensor(minus, I, I, I, I)
    
    psn = psn.tidyup()
    """remove the normalization here after adding all the possible losses in cavity interaction"""
    #print(psn.norm())
    if psn.norm() != 0:
        psn = psn.unit()
        psn = psn.tidyup()
        psn = psn.unit()
    else:
        psn = psn.tidyup()
    #print(psn.norm())
    (mea_output, sn) = measure(psn, [pz0, pz1])
    
    return mea_output, sn


"""
******************************************************
******************************************************
Functions for erroneous cnots and swap
******************************************************
******************************************************
"""

def err_cenotn(gate_fid):
    cnot = tensor(ketbra0, I)+tensor(ketbra1, X)
    cnot_bar = tensor(ketbra0, X)+tensor(ketbra1, I)
    err_cnot =  (gate_fid*tensor(I,I) + (1-gate_fid)*cnot_bar)*cnot
    
    return err_cnot

def err_cnnote(gate_fid):
    cnot = ([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])
    cnot = Qobj(cnot, dims = [[2,2],[2,2]])
    cnot_bar = ([[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
    cnot_bar = Qobj(cnot_bar, dims = [[2,2],[2,2]])
    err_cnot =  (gate_fid*tensor(I,I) + (1-gate_fid)*cnot_bar)*cnot
    
    return err_cnot

"""function for swapping erroneous swaps with different cnots"""
def swap_en(cnot_fid_en, cnot_fid_ne):
    swap = err_cenotn(cnot_fid_en)*err_cnnote(cnot_fid_ne)*err_cenotn(cnot_fid_en)
    
    return swap

"""
swap en a and b are for swapping the states of electron and nu 
at alice and bobs nodes

--input--
the cnot fidelities of e and n as controls

--returns--
swap operator for alice and bob
"""
def swap_en_a(cnot_fid_en, cnot_fid_ne):
    swap_a = tensor(I,swap_en(cnot_fid_en, cnot_fid_ne),I,I)
    
    return swap_a

def swap_en_b(cnot_fid_en, cnot_fid_ne):
    swap_b = tensor(I,I,I,swap_en(cnot_fid_en, cnot_fid_ne))
    
    return swap_b

"""
******************************************************
******************************************************
Functions for SQE
******************************************************
******************************************************
"""

"""
erroneours Single Qubit Gate

--input--
Perfect gate, error direction, error angle

--returns--
error gate
"""
def errored_sqg(perf_gate, error_dir_vector, error_angle):
    rot_matrix = I*np.cos(rotation_angle/2) - 1j*(rotation_dir_vector[0]*X+rotation_dir_vector[1]*Y+rotation_dir_vector[2]*Z)*np.sin(rotation_angle/2)
    err_gate = rot_matrix*perf_gate
    
    return err_gate

"""
******************************************************
******************************************************
Functions for spin dephasing
******************************************************
******************************************************
"""

"""
--input--
T2, time since the start

--returns--
the dephasing matrices
"""
def dephase_spin_matrices(dephasing_time, time):
    factor = np.sqrt((1-np.exp(-1*time/dephasing_time))/2)
    
    a0 = factor*I
    a1 = factor*Z
    
    return a0, a1

def sp_dephasing_a(psn_dm, dephasing_time, time):
    a0, a1 = dephase_spin_matrices(dephasing_time, time)
    a0 = tensor(I, a0, I, I, I)
    a1 = tensor(I, a1, I, I, I)
    psn_dephased = a0*psn_dm*a0.dag()+a1*psn_dm*a1.dag()
    
    return psn_dephased

def sp_dephasing_b(psn_dm, dephasing_time, time):
    a0, a1 = dephase_spin_matrices(dephasing_time, time)
    a0 = tensor(I, I, I, a0, I)
    a1 = tensor(I, I, I, a1, I)
    psn_dephased = a0*psn_dm*a0.dag()+a1*psn_dm*a1.dag()
    
    return psn_dephased

def nu_dephasing_a(psn_dm, dephasing_time, time):
    a0, a1 = dephase_spin_matrices(dephasing_time, time)
    a0 = tensor(I, I, a0, I, I)
    a1 = tensor(I, I, a1, I, I)
    psn_dephased = a0*psn_dm*a0.dag()+a1*psn_dm*a1.dag()
    
    return psn_dephased

def nu_dephasing_b(psn_dm, dephasing_time, time):
    a0, a1 = dephase_spin_matrices(dephasing_time, time)
    a0 = tensor(I, I, I, I, a0)
    a1 = tensor(I, I, I, I, a1)
    psn_dephased = a0*psn_dm*a0.dag()+a1*psn_dm*a1.dag()
    
    return psn_dephased

"""
******************************************************
******************************************************
Spin photon interaction functions in state form
******************************************************
******************************************************
"""

"""functions for entanglement with alices spins"""
def early_scat_alice_st(psn, r_vect):
    reflect_operator = r_vect[0]*tensor(ketbra0, ketbra0)+r_vect[1]*tensor(ketbra0, ketbra1)+r_vect[2]*tensor(ketbra1, ketbra0)+r_vect[3]*tensor(ketbra1, ketbra1)
    es = tensor(ketbra0, reflect_operator, I, I)+tensor(ketbra1, I, I, I, I)
    
    return es*psn

def late_scat_alice_st(psn, r_vect):
    reflect_operator = r_vect[0]*tensor(ketbra0, ketbra0)+r_vect[1]*tensor(ketbra0, ketbra1)+r_vect[2]*tensor(ketbra1, ketbra0)+r_vect[3]*tensor(ketbra1, ketbra1)
    ls = tensor(ketbra0, I, I, I, I)+tensor(ketbra1, reflect_operator, I, I)
    
    return ls*psn


def photon_spin_ent_alice_st(psn, r_vect):
    psn = early_scat_alice_st(psn, r_vect)
    psn = tensor(I, H, I, I, I)*psn
    psn = late_scat_alice_st(psn, r_vect)
    psn = tensor(I, H, I, I, I)*psn
    
    return psn

"""functions for entanglement with alices spins"""
def early_scat_bob_st(psn, r_vect):
    reflect_operator = r_vect[0]*tensor(ketbra0, ketbra0)+r_vect[1]*tensor(ketbra0, ketbra1)+r_vect[2]*tensor(ketbra1, ketbra0)+r_vect[3]*tensor(ketbra1, ketbra1)
    es = tensor(ketbra0, I, I, reflect_operator)+tensor(ketbra1, I, I, I, I)
    
    return es*psn

def late_scat_bob_st(psn, r_vect):
    reflect_operator = r_vect[0]*tensor(ketbra0, ketbra0)+r_vect[1]*tensor(ketbra0, ketbra1)+r_vect[2]*tensor(ketbra1, ketbra0)+r_vect[3]*tensor(ketbra1, ketbra1)
    ls = tensor(ketbra0, I, I, I, I)+tensor(ketbra1, I, I, reflect_operator)
    
    return ls*psn


def photon_spin_ent_bob_st(psn, r_vect):
    psn = early_scat_bob_st(psn, r_vect)
    psn = tensor(I, I, I, H, I)*psn
    psn = late_scat_bob_st(psn, r_vect)
    psn = tensor(I, I, I, H, I)*psn
    
    return psn


""""entangling the the electron spins of alice and bob
returns the entangled state with photon in |0> state and 
electrons entangled"""
def entangle_spins_st(psn, r_vect):
    
    psn = tensor(H,I,I,I,I)*psn
    
    """photon entangles with alice's spin"""
    psn_a = photon_spin_ent_alice_st(psn, r_vect)
    
    """photon entangles with bob's spin"""
    psn_ab = photon_spin_ent_bob_st(psn_a, r_vect)
    
    """measure the photon and set it to basis(2,0) using single qubit gates
        and correct the state of the spins using Z gate"""
    measure_output, psn_output = measure_photon_x(psn_ab)
    psn_output = tensor(H,I,I,I,I)*psn_output
    if measure_output == 1:
        psn_output = tensor(X,I,I,Z,I)*psn_output
    
    return psn_output



"""
******************************************************
******************************************************
Spin photon interaction functions in densiy matriix form
******************************************************
******************************************************
"""

"""
functions for entanglement with alice's spin

--input--
density matrix and reflection coeff vector:
[r00, r01, r10, r11]

--returns--
ideally, two bell states of sp and two for nu spins
"""
def early_scat_alice(psn_dm, r_vect):
    reflect_operator = r_vect[0]*tensor(ketbra0, ketbra0)+r_vect[1]*tensor(ketbra0, ketbra1)+r_vect[2]*tensor(ketbra1, ketbra0)+r_vect[3]*tensor(ketbra1, ketbra1)
    es = tensor(ketbra0, reflect_operator, I, I)+tensor(ketbra1, I, I, I, I)
    
    output_dm = es*psn_dm*es.dag()
    
    return output_dm

def late_scat_alice(psn_dm, r_vect):
    reflect_operator = r_vect[0]*tensor(ketbra0, ketbra0)+r_vect[1]*tensor(ketbra0, ketbra1)+r_vect[2]*tensor(ketbra1, ketbra0)+r_vect[3]*tensor(ketbra1, ketbra1)
    ls = tensor(ketbra0, I, I, I, I)+tensor(ketbra1, reflect_operator, I, I)
    
    output_dm = ls*psn_dm*ls.dag()
    
    return output_dm


def photon_spin_ent_alice(psn_dm, r_vect):
    psn_dm = early_scat_alice(psn_dm, r_vect)
    psn_dm = tensor(I, H, I, I, I)*psn_dm*(tensor(I, H, I, I, I)).dag()
    
    psn_dm = late_scat_alice(psn_dm, r_vect)
    psn_dm = tensor(I, H, I, I, I)*psn_dm*(tensor(I, H, I, I, I)).dag()
    
    return psn_dm

"""functions for entanglement with bob's spin"""
def early_scat_bob(psn_dm, r_vect):
    reflect_operator = r_vect[0]*tensor(ketbra0, ketbra0)+r_vect[1]*tensor(ketbra0, ketbra1)+r_vect[2]*tensor(ketbra1, ketbra0)+r_vect[3]*tensor(ketbra1, ketbra1)
    es = tensor(ketbra0, I, I, reflect_operator)+tensor(ketbra1, I, I, I, I)
    
    output_dm = es*psn_dm*es.dag()
    
    return output_dm

def late_scat_bob(psn_dm, r_vect):
    reflect_operator = r_vect[0]*tensor(ketbra0, ketbra0)+r_vect[1]*tensor(ketbra0, ketbra1)+r_vect[2]*tensor(ketbra1, ketbra0)+r_vect[3]*tensor(ketbra1, ketbra1)
    ls = tensor(ketbra0, I, I, I, I)+tensor(ketbra1, I, I, reflect_operator)
    
    output_dm = ls*psn_dm*ls.dag()
    
    return output_dm


def photon_spin_ent_bob(psn_dm, r_vect):
    psn_dm = early_scat_bob(psn_dm, r_vect)
    psn_dm = tensor(I, I, I, H, I)*psn_dm*(tensor(I, I, I, H, I)).dag()
    
    psn_dm = late_scat_bob(psn_dm, r_vect)
    psn_dm = tensor(I, I, I, H, I)*psn_dm*(tensor(I, I, I, H, I)).dag()
    
    return psn_dm
    
"""
--input--
a density matrix with photon in |0> state and the other spins
in arbitrary states.

--returns--
the entangled state  of spins with photon in |0> state and 
electrons entangled
"""
def entangle_spins_dm(psn_dm, r_vect):
    """photon entangles with alice's spin"""
    psn_dm = tensor(H,I,I,I,I)*psn_dm*(tensor(H,I,I,I,I)).dag()
    
    psn_dm_a = photon_spin_ent_alice(psn_dm, r_vect)
    
    """photon entangles with bob's spin"""
    psn_dm_ab = photon_spin_ent_bob(psn_dm_a, r_vect)
    print(psn_dm_ab.norm())
    """measure the photon and set it to basis(2,0) using single qubit gates
        and correct the state of the spins using Z gate"""
    measure_output, psn_output_dm = measure_photon_x(psn_dm_ab)
    psn_output_dm = tensor(H,I,I,I,I)*psn_output_dm*(tensor(H,I,I,I,I)).dag()
    if measure_output == 1:
        psn_output_dm = tensor(X,I,I,Z,I)*psn_output_dm*(tensor(X,I,I,Z,I)).dag()
    
    return psn_output_dm


"""
******************************************************
******************************************************
State preparation functions

three steps: 
1)sp-ph
2)swap of sp-nu
3)sp-ph
all this has to be in the denstiy matrix formalism

--input--
the cnot fidelities and the density matrix with ph in 0 state

--returns--
ideally, two bell states of sp and two for nu spins
******************************************************
******************************************************
"""

"""in state form"""
def prepare_state_st(psn_st, cnot_fid, r_vect):
    psn_st = entangle_spins_st(psn_st,r_vect)
    
    swap_a = swap_en_a(cnot_fid[0], cnot_fid[1])
    psn_st = swap_a*psn_st
    
    swap_b = swap_en_b(cnot_fid[2], cnot_fid[3])
    psn_st = swap_b*psn_st
    
    psn_st = entangle_spins_st(psn_st, r_vect)
    
    return psn_st

"""in density matrix form"""
def prepare_state_dm(psn_dm, cnot_fid, r_vect):
    psn_dm = entangle_spins_dm(psn_dm,r_vect)
    
    swap_a = swap_en_a(cnot_fid[0], cnot_fid[1])
    psn_dm = swap_a*psn_dm*swap_a.dag()
    
    swap_b = swap_en_b(cnot_fid[2], cnot_fid[3])
    psn_dm = swap_b*psn_dm*swap_b.dag()
    
    psn_dm = entangle_spins_dm(psn_dm, r_vect)
    
    return psn_dm

"""
******************************************************
******************************************************
Function for refection coefficients
******************************************************
******************************************************
"""

def reflection_coeff_sp(var_array, loss_coeff):
    ka, k, w = (1-loss_coeff)*var_array[0], var_array[0], var_array[1]
    g, gamma0, gamma1 = var_array[2], var_array[3], var_array[4]
    d, d1 = var_array[5], var_array[6]
    c0 = g**2/k/gamma0
    c1 = g**2/k/gamma1
    #print(c0)
    d0 = -d +d1
    r0 = 1 - (ka/k)/(-1j*w/k + 1/2 + c0/(-(w+d0)*1j/gamma0 + 1/2))
    r1 = 1 - (ka/k)/(-1j*w/k + 1/2 + c1/(-(w+d1)*1j/gamma1 + 1/2))
    
    return r0, r1

def reflection_coeff_nu(var_array, loss_coeff):
    ka, k, w = (1-loss_coeff)*var_array[0], var_array[0], var_array[1]
    g, gamma0, gamma1 = var_array[2], var_array[3], var_array[4]
    d, d1 = var_array[5], var_array[6]
    c0 = g**2/k/gamma0
    c1 = g**2/k/gamma1
    #print(c0)
    d0 = -d +d1
    r0 = 1 - (ka/k)/(-1j*w/k + 1/2 + c0/(-(w+d0)*1j/gamma0 + 1/2))
    r1 = 1 - (ka/k)/(-1j*w/k + 1/2 + c1/(-(w+d1)*1j/gamma1 + 1/2))
    
    return r0, r1

"""
******************************************************
******************************************************
Convert to schmidt basis

--input--
a psn state with photon in |0> 

--returns--
the state in schmidt basis
matrices for conversion to schmidt basis
******************************************************
******************************************************
"""

def basis2schmidt(psn):

    psn_data = psn.full()
    sn_data = np.array_split(psn_data, 2)[0]
    sn_matrix_form = np.reshape(sn_data, (4,4))
    
    U, S, Vh = np.linalg.svd(sn_matrix_form, full_matrices=True)
    #print(np.allclose(sn_matrix_form, np.dot(U* S, Vh)))
    
    return Qobj(S), Qobj(U).dag(), Qobj(Vh)#Qobj(S), Qobj(U, dims = [[2,2], [2,2]]).dag(), Qobj(Vh, dims = [[2,2], [2,2]]).dag()



def non_catalytic_conversion_test(prepared_dm):

    output_state = [0.5, 0.5]
    carbon_dm = ket2dm(tensor(basis(2,0), basis(2,0)))
    psnc_dm = tensor(carbon_dm, prepared_dm)
    psnc_dm = psnc_dm.permute([2,3,4,0,5,6,1])

    s_coeff, prepared_dm_schmidt_basis, pure_st = schmidt_decomp_of_dm(psnc_dm)
    input_state_array = np.reshape(np.real(s_coeff), [8])
    if np.sum(input_state_array) != 0:
        input_state_array = input_state_array/np.sum(input_state_array)
    else:
        raise Exception ("values in input state array sum to zero")
    output_state_array = concat_zeros(output_state, input_state_array)

    zero = basis(2,0)
    one = basis(2,1)
    output_state_qobj = 1/np.sqrt(2)*(tensor(zero,zero,zero,zero,zero,zero,zero) +
                                     tensor(zero,zero,zero,one,zero,zero,one))


    """use function for gamma to find the ideal gamma that is needed"""
    gamma_ideal = func_for_gamma(output_state_array, input_state_array)
    if np.sum(gamma_ideal) != 0:
        gamma_ideal = gamma_ideal/np.sum(gamma_ideal)
    else:
        raise Exception ("gamma ideal sums to zero")

    #if np.any(input_state_array == 0):
    #    print(input_state_array, "input nc")
    #    print(gamma_ideal, "gamma nc")


    """use function for locc povms to find the ideal povms to get to gamma"""
    povm_out_list, prob_out_list, perm_out_list = locc_povm_func(gamma_ideal, input_state_array)
    #print(len(povm_out_list))
    """use function for slocc povms to find the ideal povms to get to final state"""
    meas_mat, waste_mat = slocc_povm_func(output_state_array, input_state_array)

    """apply those povm on the density matrix"""
    gamma_density_mat, prob_obtained_list = locc_povm_on_dm_qobj(prepared_dm_schmidt_basis, povm_out_list, perm_out_list)

    if len(prob_obtained_list) == 0:
        majorisation_check = check_majorisation(gamma_ideal, input_state_array)
        raise Exception ("gamma failed to obtained")

    """for gamma density matrix, apply the slocc povm to get output states"""
    probab = 0
    output_density_mat, probab = slocc_povm_on_dm_qobj(gamma_density_mat, meas_mat)

    fid = fidelity(output_state_qobj, output_density_mat)
    return fid, np.real(probab), output_density_mat


def catalytic_conversion_reuse1(prepared_dm, carbon_dm_input):
    flagg = 0
    s_coeff, prepared_dm_schmidt_basis, pure_st = schmidt_decomp_of_dm(prepared_dm)
    s_coeff = np.reshape(s_coeff, [4])

    """finding and making the catalyst for the pure statestate"""
    output_states = [0.5, 0.5]
    print("sdf")
    cat_st, prob = closest_pure_state(carbon_dm_input)
    cat_st_matrix_form = np.reshape(cat_st, (2, 2))
    U, S, Vh = np.linalg.svd(cat_st_matrix_form, full_matrices=True)
    s_coeff_cat, u, v = Qobj(S), Qobj(U).dag(), Qobj(Vh).dag()
    #print(s_coeff_cat, u, v)
    basis_matrix = tensor(u, v.trans()).full()
    #print(basis_matrix)
    basis_matrix = Qobj(basis_matrix, dims = [[2,2], [2,2]])
    #print(cat_st)
    psn_schmidt_basis = (basis_matrix*cat_st).tidyup()

    s_coeff_cat = np.real((s_coeff_cat.full())**2)



    psnc_dm = tensor(carbon_dm_input, prepared_dm_schmidt_basis)

    psnc_dm = psnc_dm.permute([2,3,4,0,5,6,1])

    psnc_s_coeff, prepared_dm_schmidt_basis, pure_st = schmidt_decomp_of_dm(psnc_dm)



    input_state_array = psnc_s_coeff
    output_state_array = np.sort(np.reshape(np.tensordot(output_states, s_coeff_cat, 0), 4))[::-1]
    output_state_array = concat_zeros(output_state_array, input_state_array)

    zero = basis(2,0)
    one = basis(2,1)
    output_state_qobj = (1/np.sqrt(2)*(tensor(zero,zero,zero,zero,zero) +
                                     tensor(zero,zero,one,zero,one)))

    output_state_qobj = tensor(cat_st, prepared_dm)
    output_state_qobj = output_state_qobj.permute([2,3,4,0,5,6,1])


    """use function for gamma to find the ideal gamma that is needed"""
    gamma_ideal = func_for_gamma(output_state_array, input_state_array)
    if np.sum(gamma_ideal) != 0:
        gamma_ideal = gamma_ideal/np.sum(gamma_ideal)
    else:
        raise Exception ("gamma ideal sums to zero")


    majorisation_check = check_majorisation(gamma_ideal, input_state_array)
    if np.any(input_state_array <= 1e-2):
        flagg = 1
        output_density_mat = tensor(carbon_dm_input, prepared_dm_schmidt_basis)
        output_density_mat = output_density_mat.permute([2,3,4,0,5,6,1])
        fid = 0
        probab = 0
        print(cat_st, "carbon")
        #return 0, 0, output_density_mat, flag
        #print(input_state_array, "input")
        #print(gamma_ideal, "gamma")
    else:
        """use function for locc povms to find the ideal povms to get to gamma"""
        povm_out_list, prob_out_list, perm_out_list = locc_povm_func(gamma_ideal, input_state_array)

        """apply those povm on the density matrix"""
        gamma_density_mat, prob_obtained_list = locc_povm_on_dm_qobj(psnc_dm, povm_out_list, perm_out_list)

        if len(prob_obtained_list) == 0:
            raise Exception ("gamma failed to obtained")

        """use function for slocc povms to find the ideal povms to get to final state"""
        meas_mat, waste_mat = slocc_povm_func(output_state_array, input_state_array)

        """for each gamma density matrix, apply the slocc povm to get output states"""
        probab = 0
        output_density_mat, probab = slocc_povm_on_dm_qobj(gamma_density_mat, meas_mat)
        fid = fidelity(output_state_qobj, output_density_mat)

    return fid, np.real(probab), output_density_mat, flagg


def prepare_state_dm_test(psn_dm, cnot_error, r_vect, l_vect, reset_count, sqe_rate):
    mea_list = []
    global H_err, X_err, Z_err
    #H_err = sq_depol_err(H, sqe_rate)
    #X_err = sq_depol_err(X, sqe_rate)
    #Z_err = sq_depol_err(Z, sqe_rate)
    #print(H, X, Z)
    #print(H.dag()*H)
    #print(H_err.dag())
    psn_noloss_final = 0
    prob_loss = psn_dm.norm()
    #print(prob_loss)
    psn_dm = psn_dm.unit()
    for i in range(reset_count+1):
        psn_noloss, psn_loss, mea_list1 = prepare_state_dm(psn_dm, cnot_error, r_vect, l_vect)
        mea_list.append(mea_list1)
        #print(psn_noloss.norm())
        psn_noloss_final += prob_loss*psn_noloss
        psn_loss_final = prob_loss*psn_loss
        prob_loss = prob_loss*psn_loss.norm()
        psn_loss = reset_spins(psn_loss)
        psn_dm = psn_loss.unit()
        #print(prob_loss)

    prob_noloss = psn_noloss_final.norm()
    psn_noloss_final = psn_noloss_final.unit()

    psn_loss = psn_loss.unit()

    total_prob = prob_noloss+prob_loss

    return prob_noloss, prob_loss, total_prob

g = 10000
gm0 = 0.1
gm1 = 0.1
delta = 10000
kappa = 100000

param = [kappa,0,g,gm0,gm1,delta,0]

ph = basis(2, 0)
spin = basis(2,0)
nu = basis(2,0)
psnn = tensor(ph, spin, nu, spin, nu)
cc = basis(2,1)

gh = ket2dm(psnn)

rr = []
a, b = reflection_coeff_sp(param, 0)
c, d = reflection_coeff_nu(param, 0)

rr = [a,c,b,d]
#print(rr)
#rr = [1,1,1,1]
#print(rr)
cnot_fidy = [1,1,1,1]
#state_dm = prepare_state_dm(gh, cnot_fidy, rr)

state = entangle_spins_st(psnn, rr)
state = swap_en_a(1,1)*swap_en_b(1,1)*state
state = entangle_spins_st(state, rr)

state_dm_check = ket2dm(state)

ss = prepare_state_st(psnn, cnot_fidy, rr)

stt = ss.ptrace([1,2,3,4])
#print(stt.tr(), ss.norm())
#print(stt)


#print(tracedist(state_dm, state_dm_check), tracedist(state_dm, ket2dm(ss)))
#ss=ss.full()


#print(basis2schmidt(ss))
state = entangle_spins_st(psnn, rr)
#print(state)

s, u, v = basis2schmidt(ss) 
s = s.tidyup()
u = u.tidyup()
v = v.tidyup()
print(s)
#print(u)
#state = state.full()
#state = np.array_split(state, 2)[0]
#print(state)
#sn_matrix_form = np.reshape(state, (4,4))

jk = tensor( I, u, v).full()
jk = Qobj(jk)

state1 = ss.full()
state1 = Qobj(state1)
gh = (jk*state1).tidyup()
print(gh)
#jkk = tensor( I, I,I,v.dag()).full()

#fg = np.matmul(jk, state)
#fg = np.matmul(jkk, fg)
#print(fg)
#print(tracedist(ss, gh))

#state = state.permute((1,2))
#print(state)
#s, u, v = basis2schmidt(state) 
#s = s.tidyup()
u = u.tidyup()
v = v.tidyup()
#print(v)
#print(u)
#print(u)
#print(s)
#u = Qobj(u, dim = [[2,2],[2,2]])
#print(ss)
#print(state.transform(tensor(I, u, I, I)))
#print(state.transform(tensor(I, I, I, v)))
#print(v*v.dag())
#print(u)
#print(v)
#print(tensor(I, u, v))
#u = u.shape((2,2))
#print(U)
#print(S)
#print(Vh)
#a  = 0
#for i in range(len(S)):
#    a += S[i]**2
#S = Qobj(S)
#print(S)
#print(S.norm())

#print(a)

#print(tensor(ketbra0, X)+tensor(ketbra1, I))
#print(tensor(ketbra0, I)+tensor(ketbra1, X))

#print(psn)
#print(swap_en_a(1,1)*psn)

#ab = ab.tidyup()
#ab = Qobj.full(ab)
#ab = ab.data
#ab = np.asarray(ab)

#i =  Qobj.full(i)
#i = i.data
#i = np.asarray(i)

#ab.tofile('data2.csv', sep = ',')


#df.to_excel(filepath, index=False)"""

abc = ss.full()
aaa = np.array_split(abc, 2)[0]
aaa_re = np.reshape(aaa, (4,4))

U, S, Vh = np.linalg.svd(aaa_re, full_matrices=True)
aba = np.matmul(np.matmul(U, S), Vh)
#print(np.allclose(aaa_re, np.dot(U* S, Vh)))


ab = np.arange(3,19)
#print(ab)
ab[3] = -1
a = np.reshape(ab, (4,4))
#print(a)
u,s,v = np.linalg.svd(a)
#print(s)
u = Qobj(u)
s = Qobj(s)
v = Qobj(v)
ab = Qobj(ab)
#print(ab)
#print(s.tidyup())
#u = tensor(u.dag(), I)
#v = tensor(I, v.dag())
x = tensor(u.dag(), v)
x = x.full()
x = Qobj(x)
#print(x)
#print(x*ab)
