#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:42:23 2023

@author: hsharma4
"""

import numpy as np
from qutip import *

import sys
import os

from base_locc import *
from base_slocc import *
from base_state_change import *
from updated_measurements import *

sys.path.append(os.path.dirname(__file__))

"""basic operators"""
X = sigmax()
Z = sigmaz()
Y = sigmay()
I = qeye(2)
H = 1/np.sqrt(2)*(X+Z)
ketbra0 = ket2dm(basis(2, 0))
ketbra1 = ket2dm(basis(2, 1))
plus = ket2dm((basis(2, 0)+basis(2, 1)).unit())
minus = ket2dm((basis(2, 0)-basis(2, 1)).unit())

X_err = sigmax()
Z_err = sigmaz()
H_err = 1/np.sqrt(2)*(X+Z)

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

def errored_sqg(perf_gate, error_dir_vector, error_angle):
    rot_matrix = I*np.cos(rotation_angle/2) - 1j*(rotation_dir_vector[0]*X+
                                                  rotation_dir_vector[1]*Y+
                                                  rotation_dir_vector[2]*Z)*np.sin(rotation_angle/2)
    err_gate = rot_matrix*perf_gate

    return err_gate

def sq_depol_err(perf_gate, error_rate):
    err_gate = np.cos(error_rate)*perf_gate + 1j*np.sin(error_rate)*I#(X+Z+X*Z)
    #err_gate = rot_matrix*perf_gate

    return err_gate
"""


"""
******************************************************
******************************************************
Functions for spin dephasing
******************************************************
******************************************************
"""

"""psn is of the form |ph>|sp_a>|nu_a>|sp_b>|nu_b>
and psn_dm is a density matrix with all the information"""


def measure_photon_x(psn):
    #pz0 = tensor(plus, I, I, I, I)
    #pz1 = tensor(minus, I, I, I, I)

    psn = psn.tidyup()
    """remove the normalization here after adding all the possible losses in cavity interaction"""
    #print(psn.norm())
    if psn.norm() != 0:
        #psn = psn.unit()
        psn = psn.tidyup()
        #psn = psn.unit()
    else:
        psn = psn.tidyup()
    #print(psn.norm())
    (mea_output, sn) = measure_updated(psn, [plus, minus], targets=[0])

    return mea_output, sn

def measure_alice_spin_z(psn):
    #pz0 = tensor(plus, I, I, I, I)
    #pz1 = tensor(minus, I, I, I, I)

    psn = psn.tidyup()
    """remove the normalization here after adding all the possible losses in cavity interaction"""
    #print(psn.norm())
    if psn.norm() != 0:
        #psn = psn.unit()
        psn = psn.tidyup()
        #psn = psn.unit()
    else:
        psn = psn.tidyup()
    #print(psn.norm())
    (mea_output, sn) = measure_updated(psn, [basis(2, 0), basis(2, 1)], targets=[1])

    return mea_output, sn

def measure_bob_spin_z(psn):
    #pz0 = tensor(plus, I, I, I, I)
    #pz1 = tensor(minus, I, I, I, I)

    psn = psn.tidyup()
    """remove the normalization here after adding all the possible losses in cavity interaction"""
    #print(psn.norm())
    if psn.norm() != 0:
        #psn = psn.unit()
        psn = psn.tidyup()
        #psn = psn.unit()
    else:
        psn = psn.tidyup()
    #print(psn.norm())
    (mea_output, sn) = measure_updated(psn, [basis(2, 0), basis(2, 1)], targets=[3])

    return mea_output, sn


"""
******************************************************
******************************************************
Functions for erroneous cnots and swap
******************************************************
******************************************************
"""

def err_cenotn(theta):
    cnot = tensor(ketbra0, I)+tensor(ketbra1, X)
    cnot_bar = tensor(ketbra0, X)+tensor(ketbra1, I)
    #print(cnot_bar)
    err_cnot =  (np.cos(theta)*tensor(I,I) - 1j*np.sin(theta)*cnot_bar)*cnot

    return err_cnot

def err_cnnote(theta):
    cnot = ([[1,0,0,0],[0,0,0,1],[0,0,1,0],[0,1,0,0]])
    cnot = Qobj(cnot, dims = [[2,2],[2,2]])
    cnot_bar = ([[0,0,1,0],[0,1,0,0],[1,0,0,0],[0,0,0,1]])
    cnot_bar = Qobj(cnot_bar, dims = [[2,2],[2,2]])
    err_cnot =  (np.cos(theta)*tensor(I,I) - 1j*np.sin(theta)*cnot_bar)*cnot

    return err_cnot

"""function for swapping erroneous swaps with different cnots"""
def swap_en(theta_en, theta_ne):
    swap = err_cenotn(theta_en)*err_cnnote(theta_ne)*err_cenotn(theta_en)

    return swap

"""
swap en a and b are for swapping the states of electron and nu
at alice and bobs nodes

--input--
the cnot fidelities of e and n as controls

--returns--
swap operator for alice and bob
"""
def swap_en_a(cnot_error_en, cnot_error_ne):
    swap_a = tensor(I,swap_en(cnot_error_en, cnot_error_ne),I,I)

    return swap_a

def swap_en_b(cnot_error_en, cnot_error_ne):
    swap_b = tensor(I,I,I,swap_en(cnot_error_en, cnot_error_ne))

    return swap_b



"""
--input--
T2, time since the start

--returns--
the dephasing matrices
"""
def dephase_spin_matrices(dephasing_time, time):
    factor = (1+np.exp(-1*time/dephasing_time))/2
    #print(factor)
    factor0 = np.sqrt(factor)
    factor1 = np.sqrt(1-factor)

    a0 = factor0*I
    a1 = factor1*Z
    #print(factor0, factor1)
    return a0, a1

def sp_dephasing_alice(psn_dm, dephasing_time, time):
    a0, a1 = dephase_spin_matrices(dephasing_time, time)
    a0 = tensor(I, a0, I, I, I)
    a1 = tensor(I, a1, I, I, I)
    psn_dephased = a0*psn_dm*a0.dag()+a1*psn_dm*a1.dag()

    return psn_dephased

def sp_dephasing_bob(psn_dm, dephasing_time, time):
    a0, a1 = dephase_spin_matrices(dephasing_time, time)
    a0 = tensor(I, I, I, a0, I)
    a1 = tensor(I, I, I, a1, I)
    psn_dephased = a0*psn_dm*a0.dag()+a1*psn_dm*a1.dag()

    return psn_dephased

def nu_dephasing_alice(psn_dm, dephasing_time, time):
    a0, a1 = dephase_spin_matrices(dephasing_time, time)
    a0 = tensor(I, I, a0, I, I)
    a1 = tensor(I, I, a1, I, I)
    psn_dephased = a0*psn_dm*a0.dag()+a1*psn_dm*a1.dag()

    return psn_dephased

def nu_dephasing_bob(psn_dm, dephasing_time, time):
    a0, a1 = dephase_spin_matrices(dephasing_time, time)
    a0 = tensor(I, I, I, I, a0)
    a1 = tensor(I, I, I, I, a1)
    psn_dephased = a0*psn_dm*a0.dag()+a1*psn_dm*a1.dag()

    return psn_dephased

"""
******************************************************
******************************************************
Functions for refection and loss coefficients
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

def loss_coeff_sp(var_array, loss_coeff):
    ka, k, w = (1-loss_coeff)*var_array[0], var_array[0], var_array[1]
    g, gamma0, gamma1 = var_array[2], var_array[3], var_array[4]
    d, d1 = var_array[5], var_array[6]
    c0 = g**2/k/gamma0
    c1 = g**2/k/gamma1
    #print(c0)
    d0 = -d +d1
    l0 = np.sqrt(ka*(k-ka))/(k*(-1j*w/k + 1/2 + c0/(-(w+d0)*1j/gamma0 + 1/2)))
    l1 = np.sqrt(ka*(k-ka))/(k*(-1j*w/k + 1/2 + c1/(-(w+d1)*1j/gamma1 + 1/2)))

    return l0, l1

def loss_coeff_nu(var_array, loss_coeff):
    ka, k, w = (1-loss_coeff)*var_array[0], var_array[0], var_array[1]
    g, gamma0, gamma1 = var_array[2], var_array[3], var_array[4]
    d, d1 = var_array[5], var_array[6]
    c0 = g**2/k/gamma0
    c1 = g**2/k/gamma1
    #print(c0)
    d0 = -d +d1
    l0 = np.sqrt(ka*(k-ka))/(k*(-1j*w/k + 1/2 + c0/(-(w+d0)*1j/gamma0 + 1/2)))
    l1 = np.sqrt(ka*(k-ka))/(k*(-1j*w/k + 1/2 + c1/(-(w+d1)*1j/gamma1 + 1/2)))

    return l0, l1

def r_vector(param_sp, param_nu, loss_coeff):
    rvec = []
    a, b = reflection_coeff_sp(param_sp, loss_coeff)
    c, d = reflection_coeff_nu(param_nu, loss_coeff)

    rvec = [a,c,b,d]

    return rvec

def l_vector(param_sp, param_nu, loss_coeff):
    lvec = []
    a, b = loss_coeff_sp(param_sp, loss_coeff)
    c, d = loss_coeff_nu(param_nu, loss_coeff)

    lvec = [a,c,b,d]

    return lvec


"""
******************************************************
******************************************************
Spin photon interaction functions in state form
******************************************************
******************************************************
"""

"""functions for entanglement with alices spins"""
def early_scat_alice_st(psn, r_vect):
    reflect_operator = (r_vect[0]*tensor(ketbra0, ketbra0)+
                        r_vect[1]*tensor(ketbra0, ketbra1)+
                        r_vect[2]*tensor(ketbra1, ketbra0)+
                        r_vect[3]*tensor(ketbra1, ketbra1)
                        )

    es = tensor(ketbra0, reflect_operator, I, I)+tensor(ketbra1, I, I, I, I)

    return es*psn

def late_scat_alice_st(psn, r_vect):
    reflect_operator = (r_vect[0]*tensor(ketbra0, ketbra0)+
                        r_vect[1]*tensor(ketbra0, ketbra1)+
                        r_vect[2]*tensor(ketbra1, ketbra0)+
                        r_vect[3]*tensor(ketbra1, ketbra1)
                        )

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
    reflect_operator = (r_vect[0]*tensor(ketbra0, ketbra0)+
                        r_vect[1]*tensor(ketbra0, ketbra1)+
                        r_vect[2]*tensor(ketbra1, ketbra0)+
                        r_vect[3]*tensor(ketbra1, ketbra1)
                        )

    es = tensor(ketbra0, I, I, reflect_operator)+tensor(ketbra1, I, I, I, I)

    return es*psn

def late_scat_bob_st(psn, r_vect):
    reflect_operator = (r_vect[0]*tensor(ketbra0, ketbra0)+
                        r_vect[1]*tensor(ketbra0, ketbra1)+
                        r_vect[2]*tensor(ketbra1, ketbra0)+
                        r_vect[3]*tensor(ketbra1, ketbra1)
                        )

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
Sp-ph interaction functions in densiy matriix form
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
    reflect_operator = (r_vect[0]*tensor(ketbra0, ketbra0)+
                        r_vect[1]*tensor(ketbra0, ketbra1)+
                        r_vect[2]*tensor(ketbra1, ketbra0)+
                        r_vect[3]*tensor(ketbra1, ketbra1)
                        )

    es = tensor(ketbra0, reflect_operator, I, I)+tensor(ketbra1, I, I, I, I)

    output_dm = es*psn_dm*es.dag()

    return output_dm

def late_scat_alice(psn_dm, r_vect):
    reflect_operator = (r_vect[0]*tensor(ketbra0, ketbra0)+
                        r_vect[1]*tensor(ketbra0, ketbra1)+
                        r_vect[2]*tensor(ketbra1, ketbra0)+
                        r_vect[3]*tensor(ketbra1, ketbra1)
                        )

    ls = tensor(ketbra0, I, I, I, I)+tensor(ketbra1, reflect_operator, I, I)

    output_dm = ls*psn_dm*ls.dag()

    return output_dm


def photon_spin_ent_alice(psn_dm, r_vect, sqe_err):
    #print(psn_dm.norm(), "before es")
    psn_dm = early_scat_alice(psn_dm, r_vect)
    #print(psn_dm.norm(), "after es")
    #print(H_err)
    psn_dm = ((1-np.sum(sqe_err))*tensor(I, H, I, I, I)*psn_dm*(tensor(I, H, I, I, I)).dag()+
                            sqe_err[0]*tensor(I, X, I, I, I)*psn_dm*(tensor(I, X, I, I, I)).dag()+
                            sqe_err[1]*tensor(I, Y, I, I, I)*psn_dm*(tensor(I, Y, I, I, I)).dag()+
                            sqe_err[2]*tensor(I, Z, I, I, I)*psn_dm*(tensor(I, Z, I, I, I)).dag()
                            )


    #print(psn_dm.norm())
    psn_dm = late_scat_alice(psn_dm, r_vect)
    #print(psn_dm.norm())
    #psn_dm = tensor(I, H_err, I, I, I)*psn_dm*(tensor(I, H_err, I, I, I)).dag()
    psn_dm = ((1-np.sum(sqe_err))*tensor(I, H, I, I, I)*psn_dm*(tensor(I, H, I, I, I)).dag()+
                            sqe_err[0]*tensor(I, X, I, I, I)*psn_dm*(tensor(I, X, I, I, I)).dag()+
                            sqe_err[1]*tensor(I, Y, I, I, I)*psn_dm*(tensor(I, Y, I, I, I)).dag()+
                            sqe_err[2]*tensor(I, Z, I, I, I)*psn_dm*(tensor(I, Z, I, I, I)).dag()
                            )
    #print(psn_dm.norm())
    #raise Exception("stop")
    return psn_dm

"""functions for entanglement with bob's spin"""
def early_scat_bob(psn_dm, r_vect):
    reflect_operator = (r_vect[0]*tensor(ketbra0, ketbra0)+
                        r_vect[1]*tensor(ketbra0, ketbra1)+
                        r_vect[2]*tensor(ketbra1, ketbra0)+
                        r_vect[3]*tensor(ketbra1, ketbra1)
                        )
    es = tensor(ketbra0, I, I, reflect_operator)+tensor(ketbra1, I, I, I, I)

    output_dm = es*psn_dm*es.dag()

    return output_dm

def late_scat_bob(psn_dm, r_vect):
    reflect_operator = (r_vect[0]*tensor(ketbra0, ketbra0)+
                        r_vect[1]*tensor(ketbra0, ketbra1)+
                        r_vect[2]*tensor(ketbra1, ketbra0)+
                        r_vect[3]*tensor(ketbra1, ketbra1)
                        )

    ls = tensor(ketbra0, I, I, I, I)+tensor(ketbra1, I, I, reflect_operator)

    output_dm = ls*psn_dm*ls.dag()

    return output_dm


def photon_spin_ent_bob(psn_dm, r_vect, sqe_err):
    psn_dm = early_scat_bob(psn_dm, r_vect)
    psn_dm = ((1-np.sum(sqe_err))*tensor(I, I, I, H, I)*psn_dm*(tensor(I, I, I, H, I)).dag()+
                            sqe_err[0]*tensor(I, I, I, X, I)*psn_dm*(tensor(I, I, I, X, I)).dag()+
                            sqe_err[1]*tensor(I, I, I, Y, I)*psn_dm*(tensor(I, I, I, Y, I)).dag()+
                            sqe_err[2]*tensor(I, I, I, Z, I)*psn_dm*(tensor(I, I, I, Z, I)).dag()
                            )

    psn_dm = late_scat_bob(psn_dm, r_vect)
    psn_dm = ((1-np.sum(sqe_err))*tensor(I, I, I, H, I)*psn_dm*(tensor(I, I, I, H, I)).dag()+
                            sqe_err[0]*tensor(I, I, I, X, I)*psn_dm*(tensor(I, I, I, X, I)).dag()+
                            sqe_err[1]*tensor(I, I, I, Y, I)*psn_dm*(tensor(I, I, I, Y, I)).dag()+
                            sqe_err[2]*tensor(I, I, I, Z, I)*psn_dm*(tensor(I, I, I, Z, I)).dag()
                            )

    return psn_dm


"""
******************************************************
******************************************************
Photon loss functions
******************************************************
******************************************************
"""

"""
spontaneous emission

--input--
a density matrix with photon in |0> state and the other spins
in arbitrary states.

--returns--
the entangled state  of spins with photon in |0> state and
electrons entangled
"""

def spon_emission_prob(r_vec, l_vec):
    r_absolute_sq = np.square(np.absolute(r_vec))
    l_absolute_sq = np.square(np.absolute(l_vec))
    prob_se = np.ones(4) - r_absolute_sq - l_absolute_sq

    for i in range(4):
        if prob_se[i]<0:
            raise Exception ("negative probability of spontaneous emission")
    prob_se = np.sqrt(prob_se)

    return prob_se


"""the dm is in |E>*rho_sp_nu_sp_nu*<E|"""
def spin_collapse_function_alice(dm, r_vec, l_vec):
    prob_se = spon_emission_prob(r_vec, l_vec)

    collapse_operator = (prob_se[0]*tensor(ketbra0, ketbra0)+
                         prob_se[1]*tensor(ketbra0, ketbra1)+
                         prob_se[2]*tensor(ketbra1, ketbra0)+
                         prob_se[3]*tensor(ketbra1, ketbra1)
                         )

    collapse_operator = tensor(I, collapse_operator, I, I)
    collapsed_dm = collapse_operator*dm*collapse_operator.dag()

    return collapsed_dm




def spon_emission_alice(input_density_mat, r_vec, l_vec, sqe_err):
    """putting the photon in |+> state"""
    #input_density_mat = tensor(H,I,I,I,I)*input_density_mat*tensor(H,I,I,I,I)

    early_dm = tensor(ketbra0, I, I, I, I)*input_density_mat*tensor(ketbra0, I, I, I, I)
    density_mat_post_es = early_scat_alice(input_density_mat, r_vec)


    late_dm = ((1-np.sum(sqe_err))*tensor(I, H, I, I, I)*density_mat_post_es*tensor(I, H, I, I, I).dag()+
           sqe_err[0]*tensor(I, X, I, I, I)*density_mat_post_es*tensor(I, X, I, I, I).dag()+
           sqe_err[1]*tensor(I, Y, I, I, I)*density_mat_post_es*tensor(I, Y, I, I, I).dag()+
           sqe_err[2]*tensor(I, Z, I, I, I)*density_mat_post_es*tensor(I, Z, I, I, I).dag()
               )
    #print(late_dm.norm())
    late_dm = tensor(ketbra1, I, I, I, I)*late_dm*tensor(ketbra1, I, I, I, I)
    late_dm = tensor(X, I, I, I, I)*late_dm*tensor(X, I, I, I, I)
    #print(late_dm.norm())
    #dm_dims = input_density_mat.dims

    early_collapse_dm = spin_collapse_function_alice(early_dm, r_vec, l_vec)
    late_collapse_dm = ((1-np.sum(sqe_err))*tensor(I, H, I, I, I)*spin_collapse_function_alice(late_dm, r_vec, l_vec)*tensor(I, H, I, I, I).dag()+
                    sqe_err[0]*tensor(I, X, I, I, I)*spin_collapse_function_alice(late_dm, r_vec, l_vec)*tensor(I, X, I, I, I).dag()+
                        sqe_err[1]*tensor(I, Y, I, I, I)*spin_collapse_function_alice(late_dm, r_vec, l_vec)*tensor(I, Y, I, I, I).dag()+
                        sqe_err[2]*tensor(I, Z, I, I, I)*spin_collapse_function_alice(late_dm, r_vec, l_vec)*tensor(I, Z, I, I, I).dag()
                        )

    final_dm = early_collapse_dm + late_collapse_dm

    return final_dm


"""the dm is in |E>*rho_sp_nu_sp_nu*<E|"""
def spin_collapse_function_bob(dm, r_vec, l_vec):
    prob_se = spon_emission_prob(r_vec, l_vec)

    collapse_operator = (prob_se[0]*tensor(ketbra0, ketbra0)+
                         prob_se[1]*tensor(ketbra0, ketbra1)+
                         prob_se[2]*tensor(ketbra1, ketbra0)+
                         prob_se[3]*tensor(ketbra1, ketbra1)
                         )

    collapse_operator = tensor(I, I, I, collapse_operator)
    collapsed_dm = collapse_operator*dm*collapse_operator.dag()

    return collapsed_dm



def spon_emission_bob(input_density_mat, r_vec, l_vec, sqe_err):

    early_dm = tensor(ketbra0, I, I, I, I)*input_density_mat*tensor(ketbra0, I, I, I, I)
    density_mat_post_es = early_scat_bob(input_density_mat, r_vec)

    late_dm = ((1-np.sum(sqe_err))*tensor(I, I, I, H, I)*density_mat_post_es*tensor(I, I, I, H, I).dag()+
       sqe_err[0]*tensor(I, I, I, X, I)*density_mat_post_es*tensor(I, I, I, X, I).dag()+
       sqe_err[1]*tensor(I, I, I, Y, I)*density_mat_post_es*tensor(I, I, I, Y, I).dag()+
       sqe_err[2]*tensor(I, I, I, Z, I)*density_mat_post_es*tensor(I, I, I, Z, I).dag()
       )

    late_dm = tensor(ketbra1, I, I, I, I)*late_dm*tensor(ketbra1, I, I, I, I)
    late_dm = tensor(X, I, I, I, I)*late_dm*tensor(X, I, I, I, I)

    #dm_dims = input_density_mat.dims

    early_collapse_dm = spin_collapse_function_bob(early_dm, r_vec, l_vec)
    late_collapse_dm = ((1-np.sum(sqe_err))*tensor(I, I, I, H, I)*spin_collapse_function_bob(late_dm, r_vec, l_vec)*tensor(I, I, I, H, I).dag()+
                        sqe_err[0]*tensor(I, I, I, X, I)*spin_collapse_function_bob(late_dm, r_vec, l_vec)*tensor(I, I, I, X, I).dag()+
                        sqe_err[1]*tensor(I, I, I, Y, I)*spin_collapse_function_bob(late_dm, r_vec, l_vec)*tensor(I, I, I, Y, I).dag()+
                        sqe_err[2]*tensor(I, I, I, Z, I)*spin_collapse_function_bob(late_dm, r_vec, l_vec)*tensor(I, I, I, Z, I).dag()
                        )

    final_dm = early_collapse_dm + late_collapse_dm

    return final_dm




"""
cavity loss

--input--
a density matrix with photon in |+> state and the other spins
in arbitrary states.

--returns--
the dephased state of spins with photon in |0> state and

"""
def cavity_loss_alice(density_mat, l_vec, sqe_err):
    loss_ph_sp_dm = photon_spin_ent_alice(density_mat, l_vec, sqe_err)
    loss_sp_dm = loss_ph_sp_dm.ptrace((1,2,3,4))
    #print(loss_sp_dm.shape)

    loss_dm = tensor(ketbra0, loss_sp_dm)

    return loss_dm

def cavity_loss_bob(density_mat, l_vec, sqe_err):
    loss_ph_sp_dm = photon_spin_ent_bob(density_mat, l_vec, sqe_err)
    loss_sp_dm = loss_ph_sp_dm.ptrace((1,2,3,4))
    #print(loss_sp_dm.shape)

    loss_dm = tensor(ketbra0, loss_sp_dm)

    return loss_dm

"""
--input--
a density matrix with photon in |0> state and the other spins
in arbitrary states.

--returns--
in case of No Loss:
    the entangled state  of spins with photon in |0> state and
    electrons entangled

in case of loss of photon,:
    the spins might or might not be entangled.
    the photon is lost, but the function returns it in |0> state

therefore the photon is always returned in |0> state.
"""
def entangle_spins_dm(psn_input_dm, r_vect, l_vect, sqe_err, dist_alice_bob):

    """putting the photon in |+> state"""
    psn_dm = tensor(H,I,I,I,I)*psn_input_dm*(tensor(H,I,I,I,I)).dag()
    #print(psn_dm.norm())
    """photon entangles with alice's spin"""
    psn_dm_a = photon_spin_ent_alice(psn_dm, r_vect, sqe_err)
    #loss_flag = 0
    #print("alice es", psn_dm_a.norm())
    #print(psn_dm_a)
    """in alice's lab photon is lost"""
    psn_dm_a_cl = cavity_loss_alice(psn_dm, l_vect, sqe_err)
    psn_dm_a_se = spon_emission_alice(psn_dm, r_vect, l_vect, sqe_err)
    #loss_flag = 1

    prob_noloss_a = psn_dm_a.norm()
    prob_cl_a = psn_dm_a_cl.norm()
    prob_se_a = psn_dm_a_se.norm()

    #print(prob_noloss_a, prob_cl_a, prob_se_a)
    #raise Exception ("dfka")
    psn_dm_a = psn_dm_a.unit()
    #psn_dm_a_cl = psn_dm_a_cl.unit()
    #psn_dm_a_se = psn_dm_a_se.unit()

    
    """in kilometers"""
    l_att = 20 
    
    eta_trans = np.exp(-1*dist_alice_bob/l_att)
    #print(1-eta_trans)
    
    psn_trans_loss = (1-eta_trans)*tensor(ketbra0, psn_dm_a.ptrace([1,2,3,4]))
    #print(psn_trans_loss.norm())
    
    
    trans_time = dist_alice_bob/(3*1e5)
    psn_dm_a = sp_dephasing_bob(psn_dm_a, 3*1e-4, trans_time)
    psn_dm_a = sp_dephasing_alice(psn_dm_a, 3*1e-4,trans_time)
    psn_dm_a = nu_dephasing_bob(psn_dm_a, 0.4, trans_time)
    psn_dm_a = nu_dephasing_alice(psn_dm_a, 0.4, trans_time)


    """photon entangles with bob's spin"""
    psn_dm_ab = photon_spin_ent_bob(psn_dm_a, r_vect, sqe_err)

    """in bob's photon is lost"""
    psn_dm_b_cl = cavity_loss_bob(psn_dm_a, l_vect, sqe_err)
    psn_dm_b_se = spon_emission_bob(psn_dm_a, r_vect, l_vect, sqe_err)
    #loss_flag = 1
    #if loss_flag ==0:

    """norm of the density matrix, for finding probability of """
    prob_noloss_ab = psn_dm_ab.norm()
    prob_cl_ab = psn_dm_b_cl.norm()
    prob_se_ab = psn_dm_b_se.norm()

    psn_dm_ab = psn_dm_ab.unit()

    """measure the photon and set it to basis(2,0) using H and X gate
        and correct the state of the spins using Z gate"""

    measure_output, psn_output_dm = measure_photon_x(psn_dm_ab)
    psn_output_dm = tensor(H,I,I,I,I)*psn_output_dm*(tensor(H,I,I,I,I)).dag()
    #print(measure_output)
    if measure_output == 1:
        psn_output_dm = tensor(X,I,I,Z,I)*psn_output_dm*(tensor(X,I,I,Z,I)).dag()
    """

    meas0 = tensor(plus, I, I, I, I)
    meas1 = tensor(minus, I, I, I, I)
    psn_output_dm = meas0*psn_dm_ab*meas0.dag() + tensor(X,I,I,Z,I)*meas1*psn_dm_ab*meas1.dag()*tensor(X,I,I,Z,I).dag()
    measure_output = 0
    """

    psn_output_dm_check = (eta_trans * prob_noloss_a * (prob_noloss_ab * psn_output_dm
                       + psn_dm_b_cl + psn_dm_b_se) + psn_dm_a_cl + psn_dm_a_se
                       + prob_noloss_a * psn_trans_loss)
    psn_noloss_dm = eta_trans * prob_noloss_a * prob_noloss_ab * psn_output_dm
    psn_loss_dm = (eta_trans * prob_noloss_a * (psn_dm_b_cl + psn_dm_b_se) +
                   prob_noloss_a * psn_trans_loss +
                   (psn_dm_a_cl + psn_dm_a_se))
    #print(psn_loss_dm.norm()+psn_noloss_dm.norm())
    return psn_noloss_dm, psn_loss_dm, measure_output

"""
******************************************************
******************************************************
Convert to schmidt basis

--input--
a psn state with photon in |0>

--returns--
the state in schmidt basis |ph>|sp_a>|n_a>|sp_b>|n_b>|c_a>|c_b>
photon is in |0> basis
matrix for conversion to schmidt basis
******************************************************
******************************************************
"""

def closest_pure_state(density_mat):

    eigen_array = density_mat.eigenstates()
    #print(eigen_array)
    eigen_states = np.asarray(eigen_array[1])
    eigen_values = np.asarray(eigen_array[0])

    max_arg = np.argmax(eigen_values)
    max_eigenvalue = eigen_values[max_arg]
    max_eigenvector = eigen_states[max_arg]
    #")
    return max_eigenvector, max_eigenvalue


def basis2schmidt(psn_st):
    psn_dims = psn_st.dims
    psn_data = psn_st.full()
    """removing the photon state, as it is already in |0>"""
    sn_data = np.array_split(psn_data, 2)[0]
    psn_length = sn_data.shape[0]
    psn_length = int(np.sqrt(psn_length))
    sn_matrix_form = np.reshape(sn_data, (psn_length, psn_length))

    U, S, Vh = np.linalg.svd(sn_matrix_form, full_matrices=True)
    #print(np.allclose(sn_matrix_form, np.dot(U* S, Vh)))
    s, u, v = Qobj(S), Qobj(U).dag(), Qobj(Vh).dag()
    basis_matrix = tensor( I, u, v.trans()).full()
    basis_matrix = Qobj(basis_matrix, dims = [psn_dims[0], psn_dims[0]])

    psn = psn_st.full()
    psn = Qobj(psn, dims = psn_dims)#
    psn_schmidt_basis = (basis_matrix*psn).tidyup()
    #psn_schmidt_basis = psn_schmidt_basis.full()
    #psn_schmidt_basis = Qobj(psn_schmidt_basis, dims = psn_dims)

    return psn_schmidt_basis, basis_matrix, s, u, v.trans()    #Qobj(S), Qobj(U, dims = [[2,2], [2,2]]).dag(), Qobj(Vh, dims = [[2,2], [2,2]]).dag()


def basis2schmidt_with_catalyst(psn_st):
    psn_dims = psn_st.dims
    psn_data = psn_st.full()
    sn_data = np.array_split(psn_data, 2)[0]
    sn_matrix_form = np.reshape(sn_data, (8,8))

    U, S, Vh = np.linalg.svd(sn_matrix_form, full_matrices=True)
    #print(np.allclose(sn_matrix_form, np.dot(U* S, Vh)))
    s, u, v = Qobj(S), Qobj(U).dag(), Qobj(Vh).dag()
    basis_matrix = tensor( I, u, v.trans()).full()
    basis_matrix = Qobj(basis_matrix)

    psn = psn_st.full()
    psn = Qobj(psn)#
    psn_schmidt_basis = (basis_matrix*psn).tidyup()
    psn_schmidt_basis = psn_schmidt_basis.full()
    psn_schmidt_basis = Qobj(psn_schmidt_basis, dims = psn_dims)

    return psn_schmidt_basis, basis_matrix, s

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
def prepare_state_st(psn_st, cnot_error, r_vect):
    psn_st = entangle_spins_st(psn_st,r_vect)

    swap_a = swap_en_a(cnot_error[0], cnot_error[1])
    psn_st = swap_a*psn_st

    swap_b = swap_en_b(cnot_error[2], cnot_error[3])
    psn_st = swap_b*psn_st

    psn_st = entangle_spins_st(psn_st, r_vect)

    return psn_st

"""in density matrix form"""
def prepare_state_dm(psn_dm, cnot_error, r_vect, l_vect, sqe_err, ab_dist):
    mea = []
    psn_noloss, psn_loss_first, mea1 = entangle_spins_dm(psn_dm, r_vect, l_vect, sqe_err, ab_dist)
    mea.append(mea1)
    prob_noloss_first = psn_noloss.norm()
    #print(psn_noloss.norm()+ psn_loss_first.norm(), "onedone")
    #print("one done")
    swap_a = swap_en_a(cnot_error[0], cnot_error[1])
    psn_noloss = swap_a*psn_noloss*swap_a.dag()
    swap_b = swap_en_b(cnot_error[2], cnot_error[3])
    psn_noloss = swap_b*psn_noloss*swap_b.dag()
    #print(psn_noloss)
    psn_noloss = psn_noloss.unit()
    psn_noloss_second, psn_loss_second, mea2 = entangle_spins_dm(psn_noloss, r_vect, l_vect, sqe_err,ab_dist)
    mea.append(mea2)
    #print(psn_noloss_second.norm()+ psn_loss_second.norm(), "two done")
    psn_final = prob_noloss_first * psn_noloss_second
    psn_loss = psn_loss_first + prob_noloss_first * psn_loss_second

    #print(psn_final.norm()+psn_loss.norm())
    #print(psn_final.norm(),psn_loss.norm())
    #print()
    return psn_final, psn_loss, mea

def reset_spins(psn_dm):
    mea_output, psn_dm = measure_alice_spin_z(psn_dm)
    if mea_output == 1:
        psn_dm = tensor(I,X,I,I,I)*psn_dm*tensor(I,X,I,I,I).dag()
    psn_dm = nu_dephasing_alice(psn_dm, 1, 4)

    mea_output, psn_dm = measure_bob_spin_z(psn_dm)
    if mea_output == 1:
        psn_dm = tensor(I,I,I,X,I)*psn_dm*tensor(I,I,I,X,I).dag()

    psn_dm = nu_dephasing_bob(psn_dm, 1, 4)

    return psn_dm

def prepare_dm_withreset(psn_dm, cnot_error, r_vect, l_vect, reset_count, sqe_rate, ab_dist):
    mea_list = []
    psn_noloss_final = 0
    prob_loss = psn_dm.norm()
    psn_dm = psn_dm.unit()
    for i in range(reset_count+1):
        psn_noloss, psn_loss, mea_list1 = prepare_state_dm(psn_dm, cnot_error, r_vect, l_vect, sqe_rate, ab_dist)
        mea_list.append(mea_list1)

        psn_noloss_final += prob_loss*psn_noloss
        psn_loss_final = prob_loss*psn_loss
        prob_loss = prob_loss*psn_loss.norm()
        psn_loss = reset_spins(psn_loss)
        if psn_loss.norm() != 0:
            #print(psn_loss)
            psn_dm = psn_loss.unit()
        else:
            break


    prob_noloss = psn_noloss_final.norm()
    psn_noloss_final = psn_noloss_final.unit()

    if psn_loss.norm() != 0:
        #print(psn_loss)
        psn_loss = psn_loss.unit()
    else:
        psn_loss = psn_loss

    #
    total_prob = prob_noloss+prob_loss

    mea = np.flip(np.reshape(np.asarray(mea_list), 2*len(mea_list)))
    mea_val = int(''.join(map(lambda mea: str(int(mea)), mea)), 2)


    if total_prob < 0 or prob_noloss < 0 or prob_loss < 0:
        raise Exception("Negative probability achieved")
    if 1-total_prob > 0.1 or 1-total_prob < -0.1:
        print(psn_dm.shape)
        print(reset_count)
        print(total_prob)
        print("prob loss", prob_loss, "prob noloss", prob_noloss)
        raise Exception("Probabilities of loss and no-loss do not sum to one")

    return psn_noloss_final, psn_loss, prob_noloss, prob_loss, mea_val
