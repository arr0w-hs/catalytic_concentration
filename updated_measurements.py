#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 14:19:27 2023

@author: hsharma4
"""
import numpy as np
import scipy as sc
import math as math
import pandas as pd
import matplotlib.pyplot as plt
from qutip import *
from qutip.measurement import measure, measurement_statistics, measure_observable
"""
******************************************************
******************************************************
Functions for measurements
******************************************************
******************************************************
"""

"""measurement of photon in X basis"""
def _verify_input(op, state):
    if not isinstance(op, Qobj):
        raise TypeError("op must be a Qobj")
    if not op.isoper:
        raise ValueError("op must be all operators or all kets")
    if not isinstance(state, Qobj):
        raise TypeError("state must be a Qobj")
    if state.isket:
        if op.dims[-1] != state.dims[0]:
            raise ValueError(
                "op and state dims should be compatible when state is a ket")
    elif state.isoper:
        if op.dims != state.dims:
            raise ValueError(
                "op and state dims should match"
                " when state is a density matrix")
    else:
        raise ValueError("state must be a ket or a density matrix")
        
def _measurement_statistics_povm_ket(state, ops):
    r"""
    Returns measurement statistics (resultant states and probabilities)
    for a measurements specified by a set of positive operator valued
    measurements on a specified ket.

    Parameters
    ----------
    state : :class:`.Qobj` (ket)
            The ket specifying the state to measure.

    ops : list of :class:`.Qobj`
      List of measurement operators :math:`M_i` (specifying a POVM such that
      :math:`E_i = M_i^\dagger M_i`).


    Returns
    -------
    collapsed_states : list of :class:`.Qobj` (kets)
        The collapsed states (kets) obtained after measuring the qubits and
        obtaining the qubit specified by the target in the state specified by
        the index.

    probabilities : list of floats
        The probability of measuring a state in a the state specified by the
        index.
    """
    probabilities = []
    collapsed_states = []

    for i, op in enumerate(ops):
        p = np.absolute((state.dag() * op.dag() * op * state)[0][0][0])
        probabilities.append(p)
        if p != 0:
            collapsed_states.append((op * state) / np.sqrt(p))
        else:
            collapsed_states.append(None)

    return collapsed_states, probabilities

def _measurement_statistics_povm_dm(density_mat, ops):
    r"""
    Returns measurement statistics (resultant states and probabilities)
    for a measurements specified by a set of positive operator valued
    measurements on a specified ket or density matrix.

    Parameters
    ----------
    state : :class:`.Qobj` (density matrix)
        The ket or density matrix specifying the state to measure.

    ops : list of :class:`.Qobj`
        List of measurement operators :math:`M_i` (specifying a POVM s.t.
        :mathm:`E_i = M_i^\dagger M_i`)

    Returns
    -------
    collapsed_states : list of :class:`.Qobj`
        The collapsed states (density matrices) obtained after measuring the
        qubits and obtaining the qubit specified by the target in the state
        specified by the index.

    probabilities : list of float
        The probability of measuring a state in a the state specified by the
        index.
    """
    probabilities = []
    collapsed_states = []

    for i, op in enumerate(ops):
        st = op * density_mat * op.dag()
        p = st.tr()
        probabilities.append(p)
        if p != 0:
            collapsed_states.append(st/p)
        else:
            collapsed_states.append(None)

    return collapsed_states, probabilities

def measurement_statistics_povm(state, ops, targets=None):
    r"""
    Returns measurement statistics (resultant states and probabilities) for a
    measurement specified by a set of positive operator valued measurements on
    a specified ket or density matrix.

    Parameters
    ----------
    state : :class:`.Qobj`
        The ket or density matrix specifying the state to measure.

    ops : list of :class:`.Qobj`
        List of measurement operators :math:`M_i` or kets.  Either:

        1. specifying a POVM s.t. :math:`E_i = M_i^\dagger M_i`
        2. projection operators if ops correspond to
           projectors (s.t. :math:`E_i = M_i^\dagger = M_i`)
        3. kets (transformed to projectors)

    targets : list of ints, optional
              Specifies a list of target "qubit" indices on which to apply
              the measurement using qutip.qip.operations.gates.expand_operator
              to expand ops into full dimension.


    Returns
    -------
    collapsed_states : list of :class:`.Qobj`
        The collapsed states obtained after measuring the qubits and obtaining
        the qubit specified by the target in the state specified by the index.

    probabilities : list of floats
        The probability of measuring a state in a the state specified by the
        index.
    """
    if all(map(lambda x: x.isket, ops)):
        ops = [op * op.dag() for op in ops]

    if targets:
        N = int(np.log2(state.shape[0]))
        ops = [expand_operator(op, N=N, targets=targets) for op in ops]

    for op in ops:
        _verify_input(op, state)

    E = [op.dag() * op for op in ops]

    is_ID = sum(E)
    if not is_ID == identity(is_ID.dims[0]):
        raise ValueError("measurement operators must sum to identity")

    if state.isket:
        return _measurement_statistics_povm_ket(state, ops)
    else:
        return _measurement_statistics_povm_dm(state, ops)
    
def measure_povm(state, ops, targets=None):
    r"""
    Perform a measurement specified by list of POVMs.

    This function simulates a POVM measurement. The measurement collapses the
    state to one of the resultant states of the measurement and returns the
    index of the operator corresponding to the collapsed state as well as the
    collapsed state.

    Parameters
    ----------
    state : :class:`.Qobj`
        The ket or density matrix specifying the state to measure.

    ops : list of :class:`.Qobj`
        List of measurement operators :math:`M_i` or kets.  Either:

        1. specifying a POVM s.t. :math:`E_i = M_i^\dagger M_i`
        2. projection operators if ops correspond to projectors (s.t.
           :math:`E_i = M_i^\dagger = M_i`)
        3. kets (transformed to projectors)

    targets : list of ints, optional
        Specifies a list of target "qubit" indices on which to apply
        the measurement using
        :func:`qutip.qip.operations.gates.expand_operator`
        to expand ``ops`` into full dimension.

    Returns
    -------
    index : float
        The resultant index of the measurement.

    state : :class:`.Qobj`
        The new state (a ket if a ket was given, otherwise a density matrix).
    """
    collapsed_states, probabilities = measurement_statistics_povm(state,
                                                                  ops, targets)
    #print(probabilities)
    #print(np.sum(probabilities))
    
    if np.sum(probabilities) != 1 and np.sum(probabilities) != 0:
        probabilities = probabilities/np.sum(probabilities)
    elif np.sum(probabilities) == 0:
        return 0, state
    
    #print(np.sum(probabilities))
    index = np.random.choice(range(len(collapsed_states)), p=probabilities)
    state = collapsed_states[index]
    #print(index)
    return index, state

def measure_updated(state, ops, targets=None):
    r"""
    A dispatch method that provides measurement results handling both
    observable style measurements and projector style measurements (POVMs and
    PVMs).

    For return signatures, please check:

    - :func:`~measure_observable` for observable measurements.
    - :func:`~measure_povm` for POVM measurements.

    Parameters
    ----------
    state : :class:`.Qobj`
        The ket or density matrix specifying the state to measure.

    ops : :class:`.Qobj` or list of :class:`.Qobj`
        - measurement observable (:class:`.Qobj`); or
        - list of measurement operators :math:`M_i` or kets (list of
          :class:`.Qobj`) Either:

          1. specifying a POVM s.t. :math:`E_i = M_i^\dagger M_i`
          2. projection operators if ops correspond to projectors (s.t.
             :math:`E_i = M_i^\dagger = M_i`)
          3. kets (transformed to projectors)

    targets : list of ints, optional
        Specifies a list of target "qubit" indices on which to apply the
        measurement using :func:`qutip.qip.operations.gates.expand_operator`
        to expand ops into full dimension.
    """
    if isinstance(ops, list):
        return measure_povm(state, ops, targets)
    else:
        return measure_observable(state, ops, targets)
