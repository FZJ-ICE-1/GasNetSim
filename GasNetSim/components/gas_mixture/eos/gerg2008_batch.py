#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
#   ******************************************************************************
#     Copyright (c) 2026.
#     Developed by Yifei Lu
#    *****************************************************************************
"""
Batched entry points for the GERG-2008 numba kernels.

These wrappers exist solely to give `numba.prange` a `@njit` enclosure it can
parallelise over. The scalar kernels in gerg2008_numba.py are unchanged; each
iteration of the prange loop calls the existing scalar function on one sample.

Usage:
    out = PropertiesGERG_batch_numba(T, P, x)
    # T: (N,) float64, P: (N,) float64, x: (N, 21) float64
    # out: (N, 20) float64, columns match PropertiesGERG_numba's tuple order
"""
import numpy as np
from numba import njit, prange, float64

from .gerg2008_numba import (
    PropertiesGERG_numba,
    DensityGERG_numba,
    MolarMassGERG_numba,
)


_N_PROPERTIES = 20


@njit(float64[:, :](float64[:], float64[:], float64[:, :]),
      parallel=True, cache=True, nogil=True)
def PropertiesGERG_batch_numba(T, P, x):
    """
    Batched full-properties evaluation.

    Inputs:
        T: (N,) temperatures (K)
        P: (N,) pressures (kPa)
        x: (N, 21) mole fractions, one row per sample

    Returns:
        (N, 20) property array. Column order matches the tuple returned by
        PropertiesGERG_numba:
          0  molar_mass    5  dPdT      10 Cp          15 JT
          1  D             6  U         11 Cv_mass     16 Kappa
          2  Z             7  H         12 Cp_mass     17 rho
          3  dPdD          8  S         13 W           18 SG
          4  d2PdD2        9  Cv        14 G           19 R_specific
    """
    N = T.shape[0]
    out = np.empty((N, _N_PROPERTIES), dtype=np.float64)
    for i in prange(N):
        p = PropertiesGERG_numba(T[i], P[i], x[i])
        out[i, 0] = p[0]
        out[i, 1] = p[1]
        out[i, 2] = p[2]
        out[i, 3] = p[3]
        out[i, 4] = p[4]
        out[i, 5] = p[5]
        out[i, 6] = p[6]
        out[i, 7] = p[7]
        out[i, 8] = p[8]
        out[i, 9] = p[9]
        out[i, 10] = p[10]
        out[i, 11] = p[11]
        out[i, 12] = p[12]
        out[i, 13] = p[13]
        out[i, 14] = p[14]
        out[i, 15] = p[15]
        out[i, 16] = p[16]
        out[i, 17] = p[17]
        out[i, 18] = p[18]
        out[i, 19] = p[19]
    return out


@njit(float64[:](float64[:], float64[:], float64[:, :]),
      parallel=True, cache=True, nogil=True)
def density_batch_numba(P, T, x):
    """
    Batched density solver. Returns (N,) densities (mol/l).
    """
    N = T.shape[0]
    D = np.empty(N, dtype=np.float64)
    for i in prange(N):
        _ierr, _herr, d = DensityGERG_numba(P[i], T[i], x[i], iFlag=0)
        D[i] = d
    return D


@njit(float64[:](float64[:, :]), parallel=True, cache=True, nogil=True)
def molar_mass_batch_numba(x):
    """
    Batched molar mass. Returns (N,) g/mol.
    """
    N = x.shape[0]
    M = np.empty(N, dtype=np.float64)
    for i in prange(N):
        M[i] = MolarMassGERG_numba(x[i])
    return M
