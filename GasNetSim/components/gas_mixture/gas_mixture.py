#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
#   ******************************************************************************
#     Copyright (c) 2024.
#     Developed by Yifei Lu
#     Last change on 8/7/24, 2:20 PM
#     Last change by yifei
#    *****************************************************************************
from collections import OrderedDict
from dataclasses import dataclass
from typing import Mapping, Union

import numpy as np

from .eos import (
    calculate_gerg2008_properties,
    convert_gerg2008_to_dictionary,
    convert_to_gerg2008_composition,
)
from .viscosity import ViscosityMethod, calculate_viscosity


CompositionInput = Union[Mapping[str, float], np.ndarray]


@dataclass(eq=False)
class GasMixture:
    """Calculated gas mixture properties."""

    pressure: float
    temperature: float
    composition: OrderedDict
    eos_composition: np.ndarray
    method: str = "GERG-2008"
    viscosity_method: str = "Herning-Zipperer"
    compressibility: float = None
    specific_gravity: float = None
    molar_mass: float = None
    density: float = None
    standard_density: float = None
    joule_thomson_coefficient: float = None
    viscosity: float = None
    heat_capacity_constant_pressure: float = None
    R_specific: float = None
    HHV_J_per_m3: float = None
    HHV_J_per_sm3: float = None
    HHV_J_per_kg: float = None
    Z: float = None
    SG: float = None
    MolarMass: float = None
    rho: float = None
    JT: float = None
    Cp: float = None
    Cv: float = None


def calculate_gas_mixture(
    pressure: float,
    temperature: float,
    composition: CompositionInput,
    method: str = "GERG-2008",
    viscosity_method: str = "Herning-Zipperer",
) -> GasMixture:
    """
    Calculate EOS-backed gas mixture properties.

    Args:
        pressure: Gas pressure [Pa].
        temperature: Gas temperature [K].
        composition: Component mole fractions as a dictionary or GERG-2008 array.
        method: EOS method. Currently only GERG-2008 is supported.
        viscosity_method: Viscosity calculation method.
    """
    if method != "GERG-2008":
        raise NotImplementedError(f"Gas mixture method {method!r} is not supported.")

    eos_composition, composition_dict = _prepare_gerg2008_composition(composition)
    eos_mixture = calculate_gerg2008_properties(
        P_Pa=pressure,
        T_K=temperature,
        composition=eos_composition,
    )

    viscosity = calculate_viscosity(
        temperature,
        pressure,
        eos_composition,
        _viscosity_method(viscosity_method),
    )

    return GasMixture(
        pressure=pressure,
        temperature=temperature,
        composition=composition_dict,
        eos_composition=eos_composition,
        method=method,
        viscosity_method=viscosity_method,
        compressibility=eos_mixture.Z,
        specific_gravity=eos_mixture.SG,
        molar_mass=eos_mixture.MolarMass,
        density=eos_mixture.rho,
        standard_density=eos_mixture.standard_density,
        joule_thomson_coefficient=eos_mixture.JT,
        viscosity=viscosity,
        heat_capacity_constant_pressure=eos_mixture.Cp,
        R_specific=eos_mixture.R_specific,
        HHV_J_per_m3=eos_mixture.HHV_J_per_m3,
        HHV_J_per_sm3=eos_mixture.HHV_J_per_sm3,
        HHV_J_per_kg=eos_mixture.HHV_J_per_kg,
        Z=eos_mixture.Z,
        SG=eos_mixture.SG,
        MolarMass=eos_mixture.MolarMass,
        rho=eos_mixture.rho,
        JT=eos_mixture.JT,
        Cp=eos_mixture.Cp,
        Cv=eos_mixture.Cv,
    )


def _prepare_gerg2008_composition(
    composition: CompositionInput,
) -> tuple[np.ndarray, OrderedDict]:
    if isinstance(composition, np.ndarray):
        eos_composition = np.array(composition, dtype=float, copy=True)
        composition_dict = convert_gerg2008_to_dictionary(eos_composition)
    else:
        composition_dict = OrderedDict(composition)
        eos_composition = convert_to_gerg2008_composition(composition_dict)

    return eos_composition, composition_dict


def _viscosity_method(method: str) -> ViscosityMethod:
    if method == "Herning-Zipperer":
        return ViscosityMethod.HERNING_ZIPPERER
    if method == "Lucas":
        return ViscosityMethod.LUCAS
    raise ValueError(f"Unsupported viscosity calculation method: {method}")
