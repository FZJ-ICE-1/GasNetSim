#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
#   ******************************************************************************
#     Copyright (c) 2024.
#     Developed by Yifei Lu
#     Last change on 11/8/24, 2:22 PM
#     Last change by yifei
#    *****************************************************************************
import cantera as ct
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_allclose
from scipy.constants import bar
from GasNetSim.components.gas_mixture.GERG2008 import *

TEMPERATURE = 298  # Kelvin
PRESSURE = 1 * bar  # Pressure in Pascals


def heating_value(fuel, gas):
    """Returns the LHV and HHV for the specified fuel"""
    gas.TP = TEMPERATURE, ct.one_atm
    gas.set_equivalence_ratio(1.0, fuel, "O2:1.0")
    h1 = gas.enthalpy_mass
    Y_fuel = gas[fuel].Y[0]

    # complete combustion products
    X_products = {
        "CO2": gas.elemental_mole_fraction("C"),
        "H2O": 0.5 * gas.elemental_mole_fraction("H"),
        "N2": 0.5 * gas.elemental_mole_fraction("N"),
    }

    water = ct.Water()
    # Set liquid water state, with vapor fraction x = 0
    water.TQ = 298, 0
    h_liquid = water.h
    # Set gaseous water state, with vapor fraction x = 1
    water.TQ = 298, 1
    h_gas = water.h

    gas.TPX = None, None, X_products
    Y_H2O = gas["H2O"].Y[0]
    h2 = gas.enthalpy_mass
    LHV = -(h2 - h1) / Y_fuel
    HHV = -(h2 - h1 + (h_liquid - h_gas) * Y_H2O) / Y_fuel
    return LHV, HHV


@pytest.fixture(scope="module")
def setup_cantera_water():
    gas = ct.Solution("gri30.yaml")
    water = ct.Water()
    # Set liquid water state, with vapor fraction x = 0
    water.TQ = TEMPERATURE, 0
    h_liquid = water.h
    # Set gaseous water state, with vapor fraction x = 1
    water.TQ = TEMPERATURE, 1
    h_gas = water.h
    return gas, h_liquid, h_gas


@pytest.fixture(scope="module")
def setup_fuels():
    fuels = ["CH4", "C2H6", "C3H8", "H2", "CO"]
    molar_density = [
        0.6484926588314163,
        1.2228248480248802,
        1.8085351993461924,
        0.04065274708618872 * 2,
        1.1308810162414191,
    ]
    return fuels, molar_density


@pytest.fixture(scope="module")
def setup_gas_comp():
    return {
        "methane": 1,
        "ethane": 1,
        "propane": 1,
        "hydrogen": 1,
        "carbon monoxide": 1,
    }


def test_heating_values_cantera(setup_cantera_water, setup_fuels):
    gas, h_liquid, h_gas = setup_cantera_water
    fuels, molar_density = setup_fuels

    LHV_cantera_mass = []
    HHV_cantera_mass = []
    LHV_cantera_vol = []
    HHV_cantera_vol = []

    for i, fuel in enumerate(fuels):
        LHV, HHV = heating_value(fuel, gas)
        LHV_cantera_mass.append(LHV)
        HHV_cantera_mass.append(HHV)
        LHV_cantera_vol.append(LHV * molar_density[i])
        HHV_cantera_vol.append(HHV * molar_density[i])

    assert len(LHV_cantera_mass) == len(fuels)
    assert len(HHV_cantera_mass) == len(fuels)

    # Printing dataframes for manual verification
    LHV_df_mass = pd.DataFrame(
        {
            "molecule": fuels,
            "cantera [LHV MJ/kg]": LHV_cantera_mass,
        }
    )
    print(LHV_df_mass)

    HHV_df_mass = pd.DataFrame(
        {
            "molecule": fuels,
            "cantera [HHV MJ/kg]": HHV_cantera_mass,
        }
    )
    print(HHV_df_mass)


def test_heating_values_gerg2008(setup_gas_comp):
    gas_comp = setup_gas_comp
    LHV_gerg2008_mass = []
    HHV_gerg2008_mass = []
    LHV_gerg2008_vol = []
    HHV_gerg2008_vol = []

    for key, value in gas_comp.items():
        x = convert_to_gerg2008_composition(OrderedDict({key: value}))
        gas_mixture = GasMixtureGERG2008(
            P_Pa=PRESSURE, T_K=TEMPERATURE, composition=x, use_numba=True
        )
        HHV = gas_mixture.HHV_J_per_kg
        LHV = gas_mixture.LHV_J_per_kg
        LHV_gerg2008_mass.append(LHV)
        HHV_gerg2008_mass.append(HHV)
        HHV = gas_mixture.HHV_J_per_m3
        LHV = gas_mixture.LHV_J_per_m3
        LHV_gerg2008_vol.append(LHV)
        HHV_gerg2008_vol.append(HHV)

    assert len(LHV_gerg2008_mass) == len(gas_comp)
    assert len(HHV_gerg2008_mass) == len(gas_comp)


def test_comparisons(setup_cantera_water, setup_fuels, setup_gas_comp):
    gas, h_liquid, h_gas = setup_cantera_water
    fuels, molar_density = setup_fuels
    gas_comp = setup_gas_comp

    # Run heating value calculations for cantera
    LHV_cantera_mass = []
    HHV_cantera_mass = []
    LHV_cantera_vol = []
    HHV_cantera_vol = []

    for i, fuel in enumerate(fuels):
        LHV, HHV = heating_value(fuel, gas)
        LHV_cantera_mass.append(LHV)
        HHV_cantera_mass.append(HHV)
        LHV_cantera_vol.append(LHV * molar_density[i])
        HHV_cantera_vol.append(HHV * molar_density[i])

    # Run heating value calculations for GERG2008
    LHV_gerg2008_mass = []
    HHV_gerg2008_mass = []
    LHV_gerg2008_vol = []
    HHV_gerg2008_vol = []

    for key, value in gas_comp.items():
        x = convert_to_gerg2008_composition(OrderedDict({key: value}))
        gas_mixture = GasMixtureGERG2008(
            P_Pa=PRESSURE, T_K=TEMPERATURE, composition=x, use_numba=True
        )
        HHV = gas_mixture.HHV_J_per_kg
        LHV = gas_mixture.LHV_J_per_kg
        LHV_gerg2008_mass.append(LHV)
        HHV_gerg2008_mass.append(HHV)
        HHV = gas_mixture.HHV_J_per_m3
        LHV = gas_mixture.LHV_J_per_m3
        LHV_gerg2008_vol.append(LHV)
        HHV_gerg2008_vol.append(HHV)

    # Comparison assertions
    assert_allclose(LHV_cantera_mass, LHV_gerg2008_mass, rtol=0.01)
    assert_allclose(HHV_cantera_mass, HHV_gerg2008_mass, rtol=0.01)
    assert_allclose(LHV_cantera_vol, LHV_gerg2008_vol, rtol=0.01)
    assert_allclose(HHV_cantera_vol, HHV_gerg2008_vol, rtol=0.01)

    cantera_list = (
        LHV_cantera_mass + HHV_cantera_mass + LHV_cantera_vol + HHV_cantera_vol
    )
    gerg2008_list = (
        LHV_gerg2008_mass + HHV_gerg2008_mass + LHV_gerg2008_vol + HHV_gerg2008_vol
    )
    assert_allclose(cantera_list, gerg2008_list, rtol=0.01)
