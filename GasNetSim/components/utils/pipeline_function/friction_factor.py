#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
#   ******************************************************************************
#     Copyright (c) 2024.
#     Developed by Yifei Lu
#     Last change on 10/28/24, 12:45 AM
#     Last change by yifei
#    *****************************************************************************
import warnings
import numpy as np
from scipy.constants import atm
from scipy.optimize import fsolve
from numba import njit, float64


LAMINAR_FLOW_THRESHOLD = 2100
AIR_DENSITY = 1.225  # Density of air at standard conditions (kg/m3)


@njit(float64(float64, float64, float64, float64), cache=True, nogil=True)
def reynolds_number(diameter, velocity, rho, viscosity):
    """
    Calculate the Reynolds number.

    :param diameter: Pipe diameter (m)
    :param velocity: Fluid velocity (m/s)
    :param rho: Fluid density (kg/m^3)
    :param viscosity: Fluid viscosity (Pa*s or kg/(m*s))
    :return: Reynolds number (dimensionless)
    """
    return (diameter * abs(velocity) * rho) / viscosity


@njit(float64(float64, float64, float64, float64, float64), cache=True, nogil=True)
def reynolds_number_simple(diameter, p, sg, q, viscosity):
    """
    A simplified method to calculate the Reynolds number based on the volumetric flow rate.

    :param diameter: Pipe diameter (m)
    :param p: Pressure (Pa)
    :param sg: Gas specific gravity
    :param q: Gas flow rate (sm3/s)
    :param viscosity: Fluid viscosity (Pa*s)
    :return: Reynolds number (dimensionless)
    """
    rho_gas = sg * AIR_DENSITY * p / atm
    re = (4.0 * q * rho_gas) / (np.pi * diameter * viscosity)
    return re


@njit(float64(float64), cache=True, nogil=True)
def hagen_poiseuille(N_re):
    """
    Friction factor in Laminar zone using Hagen-Poiseuille method.

    :param N_re: Reynolds number (dimensionless)
    :return: Friction factor (dimensionless)
    """
    if N_re == 0:
        return np.inf
    return 64 / N_re


@njit(float64(float64, float64), cache=True, nogil=True)
def nikuradse(d, epsilon):
    """
    Calculate friction factor using Nikuradse method.

    :param d: Pipe diameter (m)
    :param epsilon: Pipe roughness (m)
    :return: Friction factor (dimensionless)
    """
    return 1 / (2 * np.log10(d / epsilon) + 1.14) ** 2


# @njit(float64(float64))
def von_karman_prandtl(N_re):
    """
    Von Karman-Prandtl friction factor calculation.

    :param N_re: Reynolds number (dimensionless)
    :return: Friction factor (dimensionless)
    """

    def func(f):
        return 2 * np.log10(N_re * np.sqrt(f)) - 0.8 - 1 / np.sqrt(f)

    f_init_guess = np.array(0.01)
    friction_factor = fsolve(func, f_init_guess)
    return friction_factor[0]


# @njit(float64(float64, float64, float64))
def colebrook_white(epsilon, d, N_re):
    """
    Colebrook-White equation for friction factor calculation.

    :param epsilon: Pipe roughness (m)
    :param d: Pipe diameter (m)
    :param N_re: Reynolds number (dimensionless)
    :return: Friction factor (dimensionless)
    """

    def func(f):
        return -2 * np.log10(
            epsilon / d / 3.71 + 2.51 / N_re / np.sqrt(f)
        ) - 1 / np.sqrt(f)

    f_init_guess = 0.01
    friction_factor = fsolve(func, f_init_guess)[0]
    return friction_factor


@njit(float64(float64, float64, float64), cache=True, nogil=True)
def colebrook_white_hofer_approximation(N_re, d, epsilon):
    """
    Hofer approximation for Colebrook-White for friction factor calculation.

    :param N_re: Reynolds number (dimensionless)
    :param d: Pipe diameter (m)
    :param epsilon: Pipe roughness (m)
    :return: Friction factor (dimensionless)
    """
    return (-2 * np.log10(4.518 / N_re * np.log10(N_re / 7) + epsilon / 3.71 / d)) ** (
        -2
    )


@njit(float64(float64, float64), cache=True, nogil=True)
def nikuradse_from_CWH(epsilon, d):
    """
    Calculate friction factor using the Hofer approximation Re -> inf.

    :param epsilon: Pipe roughness (m)
    :param d: Pipe diameter (m)
    :return: Friction factor (dimensionless)
    """
    return (-2 * np.log10(epsilon / (3.71 * d))) ** (-2)


@njit(float64(float64, float64, float64), cache=True, nogil=True)
def chen(epsilon, d, N_re):
    """
    Calculate friction factor using the Chen equation.

    :param epsilon: Pipe roughness (m)
    :param d: Pipe diameter (m)
    :param N_re: Reynolds number (dimensionless)
    :return: Friction factor (dimensionless)
    """
    _term1 = epsilon / d / 3.7065
    _term2 = (5.0452 / N_re) * np.log10(
        ((epsilon / d) ** 1.1098 / 2.8257) + (5.8506 / N_re) ** 0.8981
    )

    # Ensure the argument for the logarithm is positive
    _term3 = -2 * np.log10(max(_term1 - _term2, 1e-10))

    _friction_factor = 1 / (_term3**2)
    return _friction_factor


@njit(float64(float64), cache=True, nogil=True)
def weymouth(d):
    """
    Weymouth friction factor calculation.

    :param d: Pipe diameter (m)
    :return: Friction factor (dimensionless)
    """
    return 0.0093902 / (d ** (1 / 3))


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from scipy.constants import bar

    from GasNetSim.components.gas_mixture import GasMixture
    from collections import OrderedDict

    gas_comp = OrderedDict(
        [
            ("methane", 0.96522),
            ("nitrogen", 0.00259),
            ("carbon dioxide", 0.00596),
            ("ethane", 0.01819),
            ("propane", 0.0046),
            ("isobutane", 0.00098),
            ("butane", 0.00101),
            ("2-methylbutane", 0.00047),
            ("pentane", 0.00032),
            ("hexane", 0.00066),
        ]
    )

    pressures = np.arange(1, 100)
    Nre_res = []
    Nre_res_simp = []

    for p in pressures:
        gas_mixture = GasMixture(
            temperature=288.15, pressure=p * bar, composition=gas_comp
        )
        gas_mix_viscosity = gas_mixture.viscosity
        gas_mix_density = gas_mixture.density
        pipe_diameter = 0.76  # m
        volumetric_flow_rate = 20  # sm3/s
        real_volumetric_flow_rate = 20 / p  # Simple conversion to m3/s
        flow_velocity = real_volumetric_flow_rate / (
            np.pi * (pipe_diameter / 2) ** 2
        )  # m/s
        gas_mix_specific_gravity = gas_mixture.specific_gravity

        Nre = reynolds_number(
            pipe_diameter, flow_velocity, gas_mix_density, gas_mix_viscosity
        )
        Nre_simple = reynolds_number_simple(
            pipe_diameter,
            p * bar,
            gas_mix_specific_gravity,
            real_volumetric_flow_rate,
            gas_mix_viscosity,
        )

        Nre_res.append(Nre)
        Nre_res_simp.append(Nre_simple)

    plt.figure()
    plt.plot(pressures, Nre_res, label="Reynolds number (detailed)")
    plt.plot(pressures, Nre_res_simp, label="Reynolds number (simplified)")
    plt.xlabel("Pressure (bar)")
    plt.ylabel("Reynolds Number")
    plt.legend()
    plt.show()
