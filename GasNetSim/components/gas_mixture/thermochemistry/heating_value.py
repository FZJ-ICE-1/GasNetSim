#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
import json
from pathlib import Path

import numpy as np
from numba import njit

from ..eos.gerg2008_constants import number_of_atoms


REF_TEMP_COMBUSTION = [0.0, 15.0, 25.0]
_enthalpy_cache = {}


def load_enthalpy_values():
    """
    Load enthalpy values from a JSON file for a specific reference temperature.

    Returns:
        dict: Dictionary mapping temperature to arrays of enthalpy values
    """
    filepath_enthalpy = Path(__file__).with_name("absolute_enthalpies.json")
    if not filepath_enthalpy.exists():
        raise FileNotFoundError(f"ERROR: Enthalpy data file not found at {filepath_enthalpy}")

    try:
        with filepath_enthalpy.open("r") as file_obj:
            enthalpy_data = json.load(file_obj)
    except json.JSONDecodeError:
        print(f"ERROR: Invalid JSON format in enthalpy file: {filepath_enthalpy}")
        print("Please check the file format and ensure it contains valid JSON.")
        return {}
    except Exception as exc:
        print(f"ERROR: Failed to read enthalpy file: {exc}")
        return {}

    compounds = [
        "methane", "nitrogen", "carbon dioxide", "ethane", "propane",
        "isobutane", "n-butane", "isopentane", "n-pentane", "n-hexane",
        "n-heptane", "n-octane", "n-nonane", "n-decane", "hydrogen",
        "oxygen", "carbon monoxide", "water (g)", "hydrogen sulfide",
        "helium", "argon", "sulfur dioxide", "water (l)",
    ]

    result = {}
    for temp in REF_TEMP_COMBUSTION:
        temp_str = str(temp)
        cache_key = (filepath_enthalpy, temp)

        if cache_key in _enthalpy_cache:
            result[temp] = _enthalpy_cache[cache_key]
            continue

        enthalpy_values = []
        for compound in compounds:
            if compound in enthalpy_data and temp_str in enthalpy_data[compound]:
                enthalpy_values.append(enthalpy_data[compound][temp_str])
            else:
                enthalpy_values.append(0.0)
                print(f"Warning: No enthalpy data for {compound} at {temp_str}C")

        enthalpy_array = np.array(enthalpy_values)
        _enthalpy_cache[cache_key] = enthalpy_array
        result[temp] = enthalpy_array

        if "water (l)" in enthalpy_data and temp_str in enthalpy_data["water (l)"]:
            _enthalpy_cache[("water (l)", temp)] = enthalpy_data["water (l)"][temp_str]

    return result


_enthalpy_values = load_enthalpy_values()


@njit
def _CalculateHeatingValuesMolar_numba_impl(comp, enthalpy_mole):
    atom_list = number_of_atoms * comp[:, np.newaxis]
    reactants_atom = np.sum(atom_list, axis=0)

    n_CO2 = reactants_atom[1]
    n_SO2 = reactants_atom[6]
    n_H2O = reactants_atom[2] / 2

    n_O = n_CO2 * 2 + n_SO2 * 2 + n_H2O
    n_O2 = n_O / 2

    reactants = np.copy(comp)
    reactants[15] = n_O2

    reactants_enthalpy_sum = 0.0
    for i in range(len(reactants)):
        if i < len(enthalpy_mole) - 1:
            reactants_enthalpy_sum += reactants[i] * enthalpy_mole[i]

    products_enthalpy_sum = (
        n_CO2 * enthalpy_mole[2]
        + n_SO2 * enthalpy_mole[21]
        + n_H2O * enthalpy_mole[17]
    )
    LHV = reactants_enthalpy_sum - products_enthalpy_sum

    hw_liq, hw_gas = enthalpy_mole[22], enthalpy_mole[17]
    HHV = LHV + (hw_gas - hw_liq) * n_H2O

    return LHV, HHV


@njit
def _CalculateHeatingValue_numba_impl(MolarMass, MolarDensity, comp, hhv, per_mass, enthalpy_mole):
    LHV, HHV = _CalculateHeatingValuesMolar_numba_impl(comp, enthalpy_mole)

    if per_mass:
        if hhv:
            return HHV / MolarMass * 1e3
        return LHV / MolarMass * 1e3

    if hhv:
        return HHV * MolarDensity * 1e3
    return LHV * MolarDensity * 1e3


def _prepare_enthalpy_inputs(comp, reference_temp):
    if reference_temp not in REF_TEMP_COMBUSTION:
        raise ValueError(
            f"Unsupported reference temperature: {reference_temp} degree Celsius. "
            f"Use one of {REF_TEMP_COMBUSTION}."
        )

    try:
        enthalpy_array = _enthalpy_values[reference_temp]
    except KeyError as exc:
        raise ValueError(f"No enthalpy data available for reference temperature {reference_temp}") from exc

    if not isinstance(comp, np.ndarray):
        comp = np.asarray(comp, dtype=np.float64)
    elif comp.dtype != np.float64:
        comp = comp.astype(np.float64)

    return comp, enthalpy_array


def CalculateHeatingValuesMolar_numba(comp, reference_temp=25.0):
    """
    Calculate lower and higher molar heating values for a gas composition.

    Returns:
        tuple[float, float]: LHV and HHV in J/mol.
    """
    comp, enthalpy_array = _prepare_enthalpy_inputs(comp, reference_temp)
    return _CalculateHeatingValuesMolar_numba_impl(comp, enthalpy_array)


def CalculateHeatingValue_numba(MolarMass, MolarDensity, comp, hhv=True, per_mass=True, reference_temp=25.0):
    """
    Calculate the heating value of a gas mixture based on its composition and other properties.
    """
    comp, enthalpy_array = _prepare_enthalpy_inputs(comp, reference_temp)

    return _CalculateHeatingValue_numba_impl(
        float(MolarMass),
        float(MolarDensity),
        comp,
        bool(hhv),
        bool(per_mass),
        enthalpy_array,
    )
