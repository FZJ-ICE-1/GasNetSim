#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
#   ******************************************************************************
#     Copyright (c) 2024.
#     Developed by Yifei Lu
#     Last change on 10/25/24, 2:33 PM
#     Last change by yifei
#    *****************************************************************************
import warnings
import numpy as np
import math
from scipy.constants import atm
from scipy.optimize import fsolve
from numba import njit, float64


LAMINAR_FLOW_THRESHOLD = 2100
AIR_DENSITY = 1.225  # Density of air at standard conditions (kg/m3)


@njit(float64(float64, float64, float64, float64))
def reynold_number(diameter, velocity, rho, viscosity):
    """
    Calculate Reynolds number

    :param diameter: pipe diameter (m)
    :param velocity: fluid velocity (m/s)
    :param rho: fluid density (kg/m3)
    :param viscosity: fluid viscosity (Pa*s or kg/(m*s))
    :return: Reynolds number (dimensionless)
    """
    return (diameter * abs(velocity) * rho) / viscosity


@njit(float64(float64, float64, float64, float64))
def reynold_number_simple(diameter, p, sg, q, viscosity):
    """
    A simplified method to calculate the Reynolds number based on the volumetric flow rate
    :param diameter: pipe diameter (m)
    :param p: pressure (Pa)
    :param sg: gas specific gravity
    :param q: gas flow rate (sm3/s)
    :param viscosity: fluid viscosity (Pa*s)
    :return:
    """
    # pb = 101.325  # kPa
    # tb = 288.15  # K
    # return 49.44 * abs(q) * sg * pb / (viscosity * diameter * tb) * (24*3600)

    # Calculate density of the gas
    rho_gas = sg * AIR_DENSITY * p / atm

    # Calculate Reynolds number
    re = (4.0 * q * rho_gas) / (math.pi * diameter * viscosity)

    return re


@njit(float64(float64))
def hagen_poiseuille(N_re):
    """
    Friction factor in Laminar zone can be calculated using Hagen-Poiseuille method
    :param N_re:
    :return:
    """
    if N_re >= 2100:
        warnings.warn(
            "You are using Hagen-Poiseuille friction model for a non-laminar flow!"
        )
    return 64 / N_re


@njit(float64(float64, float64))
def nikuradse(d, epsilon):
    # d *= 1000
    return 1 / (2 * math.log(d / epsilon, 10) + 1.14) ** 2


@njit(float64(float64))
def von_karman_prandtl(N_re):
    """

    :param N_re: Reynolds number
    :return: von Karman - Prandtl friction factor
    """

    def func(f):
        return 2 * math.log(N_re * math.sqrt(f), 10) - 0.8 - 1 / math.sqrt(f)

    f_init_guess = np.array(0.01)
    friction_factor = fsolve(func, f_init_guess)
    return friction_factor


@njit(float64(float64, float64, float64))
def colebrook_white(epsilon, d, N_re):
    """

    :param epsilon:
    :param d:
    :param N_re:
    :return:
    """

    # d *= 1000
    def func(f):
        return -2 * np.log(epsilon / d / 3.71 + 2.51 / N_re / np.sqrt(f)) / np.log(
            10
        ) - 1 / np.sqrt(f)

    f_init_guess = 0.01
    friction_factor = fsolve(func, f_init_guess)[0]
    return friction_factor


@njit(float64(float64, float64, float64))
def colebrook_white_hofer_approximation(N_re, d, epsilon):
    return (
        -2 * math.log(4.518 / N_re * math.log(N_re / 7, 10) + epsilon / 3.71 / d, 10)
    ) ** (-2)


@njit(float64(float64, float64))
def nikuradse_from_CWH(epsilon, d):
    return (-2 * math.log(epsilon / 3.71 / d)) ** (-2)


# def chen(epsilon, d, N_re):
#     d *= 1000
#     _ = epsilon/d/3.7065 - 5.0452/N_re * math.log((((epsilon/d)**1.1096)/2.8257 + (7.149/N_re)**0.8961), 10)
#     return 1/(4 * math.log(_, 10) ** 2)


@njit(float64(float64, float64, float64))
def chen(epsilon, d, N_re):
    # Calculate the intermediate values according to the Chen equation
    _term1 = epsilon / d / 3.7065
    _term2 = (
        5.0452
        / N_re
        * math.log10(((epsilon / d) ** 1.1098 / 2.8257) + (5.8506 / N_re) ** 0.8981)
    )

    # Ensure the argument for the logarithm is positive
    if _term1 - _term2 <= 0:
        raise ValueError(
            "Logarithm argument must be positive. Check the Reynolds number or flow velocity!"
        )

    _term3 = -2 * math.log10(_term1 - _term2)

    _friction_factor = 1 / (_term3**2)
    return _friction_factor


@njit(float64(float64))
def weymouth(d):
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

    Nre_res = []
    Nre_res_simp = []

    for p in range(1, 100):
        gas_mixture = GasMixture(
            temperature=288.15, pressure=p * bar, composition=gas_comp
        )

        gas_mix_viscosity = gas_mixture.viscosity
        gas_mix_density = gas_mixture.density
        pipe_diameter = 0.76  # m
        volumetric_flow_rate = 20  # sm3/s
        real_volumetric_flow_rate = 20 / p  # Simple conversion to m3/s
        flow_velocity = real_volumetric_flow_rate / (
            math.pi * (pipe_diameter / 2) ** 2
        )  # m/s
        gas_mix_specific_gravity = gas_mixture.specific_gravity

        Nre = reynold_number(
            pipe_diameter, flow_velocity, gas_mix_density, gas_mix_viscosity
        )
        Nre_simple = reynold_number_simple(
            pipe_diameter,
            p * bar,
            gas_mix_specific_gravity,
            real_volumetric_flow_rate,
            gas_mix_viscosity,
        )

        Nre_res.append(Nre)
        Nre_res_simp.append(Nre_simple)

    plt.figure()
    plt.plot(Nre_res, label="Reynolds number (detailed)")
    plt.plot(Nre_res_simp, label="Reynolds number (simplified)")
    plt.xlabel("Pressure (bar)")
    plt.ylabel("Reynolds Number")
    plt.legend()
    plt.show()
