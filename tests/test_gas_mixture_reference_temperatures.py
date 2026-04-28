from collections import OrderedDict

from numpy.testing import assert_allclose
from scipy.constants import bar

from GasNetSim.components.gas_mixture import calculate_gas_mixture


def test_density_and_combustion_reference_temperatures_are_independent():
    composition = OrderedDict({"methane": 0.9, "hydrogen": 0.1})

    default_25_0 = calculate_gas_mixture(
        pressure=50 * bar,
        temperature=300.0,
        composition=composition,
    )
    same_comb_25_15 = calculate_gas_mixture(
        pressure=50 * bar,
        temperature=300.0,
        composition=composition,
        T_ref_dens_degreeC=15.0,
        T_ref_comb_degreeC=25.0,
    )
    same_density_15_0 = calculate_gas_mixture(
        pressure=50 * bar,
        temperature=300.0,
        composition=composition,
        T_ref_dens_degreeC=0.0,
        T_ref_comb_degreeC=15.0,
    )

    assert_allclose(default_25_0.HHV_J_per_kg, same_comb_25_15.HHV_J_per_kg)
    assert_allclose(default_25_0.HHV_J_per_m3, same_comb_25_15.HHV_J_per_m3)
    assert default_25_0.standard_density != same_comb_25_15.standard_density
    assert default_25_0.HHV_J_per_sm3 != same_comb_25_15.HHV_J_per_sm3

    assert_allclose(default_25_0.standard_density, same_density_15_0.standard_density)
    assert default_25_0.HHV_J_per_kg != same_density_15_0.HHV_J_per_kg
    assert default_25_0.HHV_J_per_m3 != same_density_15_0.HHV_J_per_m3
