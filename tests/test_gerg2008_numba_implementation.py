#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
#   ******************************************************************************
#     Copyright (c) 2025.
#     Developed by Yifei Lu
#     Last change on 1/6/25, 3:53 PM
#     Last change by yifei
#    *****************************************************************************

# **********************************************************************************************************************
# This script contains a comprehensive set of unit tests designed to validate the functionalities and calculations
# within the GasNetSim components. The tests primarily focus on the GasMixtureGERG2008 class found in the gas_mixture
# module of GasNetSim. These tests aim to ensure accurate and reliable performance of critical methods related to gas
# mixture properties and calculations.

# The script includes test cases for various functions within the GasMixtureGERG2008 class, such as the hyperbolic
# tangent, hyperbolic sine, and hyperbolic cosine functions, as well as methods like CalculateHeatingValue,
# ConvertCompositionGERG, MolarMassGERG, PressureGERG, DensityGERG, Alpha0GERG, ReducingParametersGERG,
# PropertiesGERG, PseudoCriticalPointGERG, and AlpharGERG.

from numpy.testing import assert_almost_equal, assert_allclose

# Each test is designed to assert the correctness and consistency of calculations involved in determining properties
# like heating value, molar mass, pressure, density, ideal gas Helmholtz energy, reducing parameters,
# pseudo-critical point, and residual Helmholtz energy.
# **********************************************************************************************************************
from scipy.constants import bar

from GasNetSim.components.gas_mixture.GERG2008 import *
from GasNetSim.components.gas_mixture.GERG2008 import convert_to_gerg2008_composition


# Test the tanh, sinh, and cosh functions
def test_tanh_sinh_cosh():
    """
    Test the numba version of the hyperbolic tangent (tanh), hyperbolic sine (sinh), and hyperbolic cosine (cosh) functions.
    """
    test_cases = [-1.0, 0.0, 1.0]
    for x in test_cases:
        assert_almost_equal(Tanh_numba(x), Tanh(x))
        assert_almost_equal(Sinh_numba(x), Sinh(x))
        assert_almost_equal(Cosh_numba(x), Cosh(x))


def test_heating_value():
    """
    Test the numba version of the CalculateHeatingValue function of GasMixtureGERG2008 class.
    """
    # Create the NIST gas mixture dictionary
    nist_gas_mixture = {}
    a = [
        "methane",
        "nitrogen",
        "carbon dioxide",
        "ethane",
        "propane",
        "isobutane",
        "butane",
        "isopentane",
        "pentane",
        "hexane",
        "heptane",
        "octane",
        "nonane",
        "decane",
        "hydrogen",
        "oxygen",
        "carbon monoxide",
        "water",
        "hydrogen sulfide",
        "helium",
        "argon",
    ]
    b = np.array(
        [
            0.77824,
            0.02,
            0.06,
            0.08,
            0.03,
            0.0015,
            0.003,
            0.0005,
            0.00165,
            0.00215,
            0.00088,
            0.00024,
            0.00015,
            0.00009,
            0.004,
            0.005,
            0.002,
            0.0001,
            0.0025,
            0.007,
            0.001,
        ]
    )
    for _i in range(21):
        nist_gas_mixture[a[_i]] = b[_i]

    gerg2008_composition = convert_to_gerg2008_composition(nist_gas_mixture)

    # Create an instance of the GasMixtureGERG2008 class with the NIST gas mixture
    gas_mixture = GasMixtureGERG2008(
        500 * bar, 400, gerg2008_composition, use_numba=False
    )

    # Test the CalculateHeatingValue function
    expected_heating_value = gas_mixture.CalculateHeatingValue(
        comp=gerg2008_composition, hhv=True, parameter="volume"
    )
    molarmass = gas_mixture.MolarMass
    molardensity = gas_mixture.MolarDensity
    calculated_heating_value = CalculateHeatingValue_numba(
        MolarMass=molarmass,
        MolarDensity=molardensity,
        comp=gerg2008_composition,
        hhv=True,
        per_mass=False,
        reference_temp=25.0,
    )

    # assert_almost_equal(calculated_heating_value, expected_heating_value, decimal=5)
    np.testing.assert_allclose(
        calculated_heating_value, expected_heating_value, rtol=1e-5
    )


# def test_convert_composition_gerg():
#     """
#     Test the numba version of the ConvertCompositionGERG method of GasMixtureGERG2008 class.
#     """
#     # Create the NIST gas mixture dictionary
#     nist_gas_mixture = {}
#
#     a = ['methane', 'nitrogen', 'carbon dioxide', 'ethane', 'propane', 'isobutane',
#          'butane', 'isopentane', 'pentane', 'hexane', 'heptane', 'octane', 'nonane',
#          'decane', 'hydrogen', 'oxygen', 'carbon monoxide', 'water', 'hydrogen sulfide',
#          'helium', 'argon']
#     b = np.array([0.77824, 0.02, 0.06, 0.08, 0.03, 0.0015, 0.003, 0.0005, 0.00165, 0.00215, 0.00088, 0.00024, 0.00015, 0.00009,
#          0.004, 0.005, 0.002, 0.0001, 0.0025, 0.007, 0.001])
#     for ii in range(21):
#         nist_gas_mixture[a[ii]] = b[ii]
#
#     nist_gas_mixture_gerg2008_composition = convert_to_gerg2008_composition(nist_gas_mixture)
#
#     # Create an instance of the GasMixtureGERG2008 class with the NIST gas mixture
#     gas_mixture = GasMixtureGERG2008(500 * bar, 400, nist_gas_mixture_gerg2008_composition, use_numba=False)
#
#     # Test the ConvertCompositionGERG function
#     expected_result = gas_mixture.x[1:]
#
#     # Calculate the converted composition using ConvertCompositionGERG method
#     converted_composition = convertCompositionGERG_numba(nist_gas_mixture)
#     assert_almost_equal(converted_composition, expected_result)


def test_molarmass_gerg():
    """
    Test the numba version of the MolarMassGERG method of GasMixtureGERG2008 class.
    """
    # Create the NIST gas mixture dictionary
    nist_gas_mixture = {}
    a = [
        "methane",
        "nitrogen",
        "carbon dioxide",
        "ethane",
        "propane",
        "isobutane",
        "butane",
        "isopentane",
        "pentane",
        "hexane",
        "heptane",
        "octane",
        "nonane",
        "decane",
        "hydrogen",
        "oxygen",
        "carbon monoxide",
        "water",
        "hydrogen sulfide",
        "helium",
        "argon",
    ]
    b = np.array(
        [
            0.77824,
            0.02,
            0.06,
            0.08,
            0.03,
            0.0015,
            0.003,
            0.0005,
            0.00165,
            0.00215,
            0.00088,
            0.00024,
            0.00015,
            0.00009,
            0.004,
            0.005,
            0.002,
            0.0001,
            0.0025,
            0.007,
            0.001,
        ]
    )
    for ii in range(21):
        nist_gas_mixture[a[ii]] = b[ii]

    nist_gas_mixture_gerg2008_composition = convert_to_gerg2008_composition(
        nist_gas_mixture
    )

    # Create an instance of the GasMixtureGERG2008 class with the NIST gas mixture
    gas_mixture = GasMixtureGERG2008(
        500 * bar, 400, nist_gas_mixture_gerg2008_composition, use_numba=False
    )

    # Calculate the expected molar mass manually based on the given mixture
    expected_molar_mass = gas_mixture.MolarMassGERG()

    # Get the calculated molar mass from the MolarMassGERG method
    calculated_molar_mass = MolarMassGERG_numba(b)
    assert_almost_equal(expected_molar_mass, calculated_molar_mass)


def test_pressure_gerg():
    """
    Test the numba version of the PressureGERG method of GasMixtureGERG2008 class.
    """
    # Create the NIST gas mixture dictionary
    nist_gas_mixture = {}
    a = [
        "methane",
        "nitrogen",
        "carbon dioxide",
        "ethane",
        "propane",
        "isobutane",
        "butane",
        "isopentane",
        "pentane",
        "hexane",
        "heptane",
        "octane",
        "nonane",
        "decane",
        "hydrogen",
        "oxygen",
        "carbon monoxide",
        "water",
        "hydrogen sulfide",
        "helium",
        "argon",
    ]
    b = np.array(
        [
            0.77824,
            0.02,
            0.06,
            0.08,
            0.03,
            0.0015,
            0.003,
            0.0005,
            0.00165,
            0.00215,
            0.00088,
            0.00024,
            0.00015,
            0.00009,
            0.004,
            0.005,
            0.002,
            0.0001,
            0.0025,
            0.007,
            0.001,
        ]
    )
    for ii in range(21):
        nist_gas_mixture[a[ii]] = b[ii]

    nist_gas_mixture_gerg2008_composition = convert_to_gerg2008_composition(
        nist_gas_mixture
    )

    # Create an instance of the GasMixtureGERG2008 class with the NIST gas mixture
    gas_mixture = GasMixtureGERG2008(
        500 * bar, 400, nist_gas_mixture_gerg2008_composition, use_numba=False
    )

    # Define the density input for PressureGERG method
    d = 10

    # Calculate the expected pressure using an example formula or method
    expected_values = gas_mixture.PressureGERG(d)

    Temp = gas_mixture.T
    AR = np.array(gas_mixture.AlpharGERG(itau=0, idelta=0, D=d))
    # Call the PressureGERG method with the given diameter
    calculated_values = PressureGERG_numba(Temp, d, b)
    assert_almost_equal(expected_values, calculated_values)


def test_density_gerg():
    """
    Test the numba version of the DensityGERG function of GasMixtureGERG2008 class.
    """
    # Create the NIST gas mixture dictionary
    nist_gas_mixture = {}
    a = [
        "methane",
        "nitrogen",
        "carbon dioxide",
        "ethane",
        "propane",
        "isobutane",
        "butane",
        "isopentane",
        "pentane",
        "hexane",
        "heptane",
        "octane",
        "nonane",
        "decane",
        "hydrogen",
        "oxygen",
        "carbon monoxide",
        "water",
        "hydrogen sulfide",
        "helium",
        "argon",
    ]
    b = np.array(
        [
            0.77824,
            0.02,
            0.06,
            0.08,
            0.03,
            0.0015,
            0.003,
            0.0005,
            0.00165,
            0.00215,
            0.00088,
            0.00024,
            0.00015,
            0.00009,
            0.004,
            0.005,
            0.002,
            0.0001,
            0.0025,
            0.007,
            0.001,
        ]
    )
    for ii in range(21):
        nist_gas_mixture[a[ii]] = b[ii]

    nist_gas_mixture_gerg2008_composition = convert_to_gerg2008_composition(
        nist_gas_mixture
    )

    # Create an instance of the GasMixtureGERG2008 class with the NIST gas mixture
    gas_mixture = GasMixtureGERG2008(
        500 * bar, 400, nist_gas_mixture_gerg2008_composition, use_numba=False
    )

    # Define the density input for PressureGERG method
    d = gas_mixture.MolarDensity

    # Expected value calculated from the function call
    # expected_values = gas_mixture.PressureGERG(d)
    _, _, expected_values = gas_mixture.DensityGERG()

    # AR = np.array(gas_mixture.AlpharGERG(itau=0, idelta=0, D=d))
    Press = gas_mixture.P
    Temp = gas_mixture.T

    # Test the DensityGERG function with iFlag=0 (default)
    _, _, calculated_values = DensityGERG_numba(
        Press, Temp, b, iFlag=0
    )  # Calling the function without any argument
    assert_allclose(expected_values, calculated_values)


def test_alpha0_gerg():
    """
    Test the numba version of the Alpha0GERG() function of GasMixtureGERG2008 class.
    """
    # Create the NIST gas mixture dictionary
    nist_gas_mixture = {}
    a = [
        "methane",
        "nitrogen",
        "carbon dioxide",
        "ethane",
        "propane",
        "isobutane",
        "butane",
        "isopentane",
        "pentane",
        "hexane",
        "heptane",
        "octane",
        "nonane",
        "decane",
        "hydrogen",
        "oxygen",
        "carbon monoxide",
        "water",
        "hydrogen sulfide",
        "helium",
        "argon",
    ]
    b = np.array(
        [
            0.77824,
            0.02,
            0.06,
            0.08,
            0.03,
            0.0015,
            0.003,
            0.0005,
            0.00165,
            0.00215,
            0.00088,
            0.00024,
            0.00015,
            0.00009,
            0.004,
            0.005,
            0.002,
            0.0001,
            0.0025,
            0.007,
            0.001,
        ]
    )
    for ii in range(21):
        nist_gas_mixture[a[ii]] = b[ii]

    nist_gas_mixture_gerg2008_composition = convert_to_gerg2008_composition(
        nist_gas_mixture
    )

    # Create an instance of the GasMixtureGERG2008 class with the NIST gas mixture
    gas_mixture = GasMixtureGERG2008(
        500 * bar, 400, nist_gas_mixture_gerg2008_composition, use_numba=False
    )

    # Expected value calculated from the function call
    # a0(0) - Ideal gas Helmholtz energy (all dimensionless [i.e., divided by RT])
    # a0(1) - tau*partial(a0)/partial(tau)
    # a0(2) - tau^2*partial^2(a0)/partial(tau)^2
    expected_alpha0 = gas_mixture.Alpha0GERG()

    Temp = gas_mixture.T
    MolarDensity = gas_mixture.MolarDensity
    X = b

    # Call the Alpha0GERG function
    actual_alpha0 = Alpha0GERG_numba(Temp, MolarDensity, X)
    assert_almost_equal(actual_alpha0, expected_alpha0)


def test_reducing_parameters_gerg():
    """
    Test the numba version of the ReducingParametersGERG() function of GasMixtureGERG2008 class.
    """
    # Create the NIST gas mixture dictionary
    nist_gas_mixture = {}
    a = [
        "methane",
        "nitrogen",
        "carbon dioxide",
        "ethane",
        "propane",
        "isobutane",
        "butane",
        "isopentane",
        "pentane",
        "hexane",
        "heptane",
        "octane",
        "nonane",
        "decane",
        "hydrogen",
        "oxygen",
        "carbon monoxide",
        "water",
        "hydrogen sulfide",
        "helium",
        "argon",
    ]
    b = np.array(
        [
            0.77824,
            0.02,
            0.06,
            0.08,
            0.03,
            0.0015,
            0.003,
            0.0005,
            0.00165,
            0.00215,
            0.00088,
            0.00024,
            0.00015,
            0.00009,
            0.004,
            0.005,
            0.002,
            0.0001,
            0.0025,
            0.007,
            0.001,
        ]
    )
    for ii in range(21):
        nist_gas_mixture[a[ii]] = b[ii]

    nist_gas_mixture_gerg2008_composition = convert_to_gerg2008_composition(
        nist_gas_mixture
    )

    # Create an instance of the GasMixtureGERG2008 class with the NIST gas mixture
    gas_mixture = GasMixtureGERG2008(
        500 * bar, 400, nist_gas_mixture_gerg2008_composition, use_numba=False
    )

    # Expected value calculated from the function call
    expected_reducingparametersgerg = gas_mixture.ReducingParametersGERG()

    # Call the ReducingParametersGERG function
    # Tr - Reducing temperature(K)
    # Dr - Reducing density(mol / l)
    actual_reducingparametersgerg = ReducingParametersGERG_numba(b)
    assert_almost_equal(actual_reducingparametersgerg, expected_reducingparametersgerg)


def test_pseudo_critical_point_gerg():
    """
    Test the numba version of the PseudoCriticalPointGERG() function of GasMixtureGERG2008 class.
    """
    # Create the NIST gas mixture dictionary
    nist_gas_mixture = {}
    a = [
        "methane",
        "nitrogen",
        "carbon dioxide",
        "ethane",
        "propane",
        "isobutane",
        "butane",
        "isopentane",
        "pentane",
        "hexane",
        "heptane",
        "octane",
        "nonane",
        "decane",
        "hydrogen",
        "oxygen",
        "carbon monoxide",
        "water",
        "hydrogen sulfide",
        "helium",
        "argon",
    ]
    b = np.array(
        [
            0.77824,
            0.02,
            0.06,
            0.08,
            0.03,
            0.0015,
            0.003,
            0.0005,
            0.00165,
            0.00215,
            0.00088,
            0.00024,
            0.00015,
            0.00009,
            0.004,
            0.005,
            0.002,
            0.0001,
            0.0025,
            0.007,
            0.001,
        ]
    )
    for ii in range(21):
        nist_gas_mixture[a[ii]] = b[ii]

    nist_gas_mixture_gerg2008_composition = convert_to_gerg2008_composition(
        nist_gas_mixture
    )

    # Create an instance of the GasMixtureGERG2008 class with the NIST gas mixture
    gas_mixture = GasMixtureGERG2008(
        500 * bar, 400, nist_gas_mixture_gerg2008_composition, use_numba=False
    )

    # Expected value calculated from the function call
    expected_pseudocriticalpointgerg = gas_mixture.PseudoCriticalPointGERG()

    # Call the ReducingParametersGERG function
    actual_pseudocriticalpointgerg = PseudoCriticalPointGERG_numba(b)
    assert_allclose(actual_pseudocriticalpointgerg, expected_pseudocriticalpointgerg)


def test_alphar_gerg():
    """
    Test the numba version of the AlpharGERG() function of GasMixtureGERG2008 class.
    """
    # Create the NIST gas mixture dictionary
    nist_gas_mixture = {}
    a = [
        "methane",
        "nitrogen",
        "carbon dioxide",
        "ethane",
        "propane",
        "isobutane",
        "butane",
        "isopentane",
        "pentane",
        "hexane",
        "heptane",
        "octane",
        "nonane",
        "decane",
        "hydrogen",
        "oxygen",
        "carbon monoxide",
        "water",
        "hydrogen sulfide",
        "helium",
        "argon",
    ]
    b = np.array(
        [
            0.77824,
            0.02,
            0.06,
            0.08,
            0.03,
            0.0015,
            0.003,
            0.0005,
            0.00165,
            0.00215,
            0.00088,
            0.00024,
            0.00015,
            0.00009,
            0.004,
            0.005,
            0.002,
            0.0001,
            0.0025,
            0.007,
            0.001,
        ]
    )
    for ii in range(21):
        nist_gas_mixture[a[ii]] = b[ii]

    nist_gas_mixture_gerg2008_composition = convert_to_gerg2008_composition(
        nist_gas_mixture
    )

    # Create an instance of the GasMixtureGERG2008 class with the NIST gas mixture
    gas_mixture = GasMixtureGERG2008(
        500 * bar, 400, nist_gas_mixture_gerg2008_composition, use_numba=False
    )

    # Expected value calculated from the function call
    #                         ar(0,0) - Residual Helmholtz energy (dimensionless, =a/RT)
    #                         ar(0,1) -     delta*partial  (ar)/partial(delta)
    #                         ar(0,2) -   delta^2*partial^2(ar)/partial(delta)^2
    #                         ar(0,3) -   delta^3*partial^3(ar)/partial(delta)^3
    #                         ar(1,0) -       tau*partial  (ar)/partial(tau)
    #                         ar(1,1) - tau*delta*partial^2(ar)/partial(tau)/partial(delta)
    #                         ar(2,0) -     tau^2*partial^2(ar)/partial(tau)^2

    D = 15.03402741629294
    expected_alphargerg = gas_mixture.AlpharGERG(1, 0, D)

    Temp = gas_mixture.T

    # Call the ReducingParametersGERG function
    actual_alphargerg = AlpharGERG_numba(Temp, b, 1, 0, D)

    assert_almost_equal(actual_alphargerg, expected_alphargerg)


def test_PropertiesGERG():
    """
    Test the numba version of the PropertiesGERG_numba() function.
    """
    # Create the NIST gas mixture dictionary
    nist_gas_mixture = {}
    a = [
        "methane",
        "nitrogen",
        "carbon dioxide",
        "ethane",
        "propane",
        "isobutane",
        "butane",
        "isopentane",
        "pentane",
        "hexane",
        "heptane",
        "octane",
        "nonane",
        "decane",
        "hydrogen",
        "oxygen",
        "carbon monoxide",
        "water",
        "hydrogen sulfide",
        "helium",
        "argon",
    ]
    b = np.array(
        [
            0.77824,
            0.02,
            0.06,
            0.08,
            0.03,
            0.0015,
            0.003,
            0.0005,
            0.00165,
            0.00215,
            0.00088,
            0.00024,
            0.00015,
            0.00009,
            0.004,
            0.005,
            0.002,
            0.0001,
            0.0025,
            0.007,
            0.001,
        ]
    )
    for ii in range(21):
        nist_gas_mixture[a[ii]] = b[ii]

    nist_gas_mixture_gerg2008_composition = convert_to_gerg2008_composition(
        nist_gas_mixture
    )

    # Create an instance of the GasMixtureGERG2008 class with the NIST gas mixture
    gas_mixture = GasMixtureGERG2008(
        500 * bar, 400, nist_gas_mixture_gerg2008_composition, use_numba=False
    )

    expected_PropertiesGERG = [
        gas_mixture.MolarMass,
        gas_mixture.MolarDensity,
        gas_mixture.Z,
        gas_mixture.dPdD,
        gas_mixture.d2PdD2,
        gas_mixture.dPdT,
        gas_mixture.energy,
        gas_mixture.enthalpy,
        gas_mixture.entropy,
        gas_mixture.Cv_molar,
        gas_mixture.Cp_molar,
        gas_mixture.Cv,
        gas_mixture.Cp,
        gas_mixture.c,
        gas_mixture.gibbs_energy,
        gas_mixture.JT,
        gas_mixture.isentropic_exponent,
        gas_mixture.rho,
        gas_mixture.SG,
        gas_mixture.R_specific,
    ]

    # Call the PropertiesGERG function
    calculated_PropertiesGERG = PropertiesGERG_numba(gas_mixture.T, gas_mixture.P, b)

    assert_allclose(expected_PropertiesGERG, calculated_PropertiesGERG)


def test_separate_properties_GERG2008():
    """
    Test function to verify that the outputs of the refactored functions match those of the original PropertiesGERG2008 function.
    """
    T = 300.0  # Temperature in K
    P = 101.325  # Pressure in kPa
    x = np.array(
        [
            0.77824,
            0.02,
            0.06,
            0.08,
            0.03,
            0.0015,
            0.003,
            0.0005,
            0.00165,
            0.00215,
            0.00088,
            0.00024,
            0.00015,
            0.00009,
            0.004,
            0.005,
            0.002,
            0.0001,
            0.0025,
            0.007,
            0.001,
        ]
    )
    # Original PropertiesGERG2008 function call (assuming it returns the expected outputs)
    original_results = PropertiesGERG_numba(T, P, x)

    # Calculate density and common properties to reuse in subsequent calculations
    D = density_numba(P, T, x)
    a0, ar = common_properties_numba(T, D, x)

    # Calculate properties using the refactored functions
    Z = compressibility_factor_numba(ar)
    refactored_results = (
        molar_mass_numba(x),
        D,
        Z,
        first_derivative_pressure_density_numba(RGERG, T, ar),
        second_derivative_pressure_temperature_density_numba(RGERG, T, D, ar),
        first_derivative_pressure_temperature_numba(RGERG, D, ar),
        internal_energy_numba(RGERG, T, a0, ar),
        enthalpy_numba(RGERG, T, a0, ar),
        entropy_numba(RGERG, a0, ar),
        isochoric_heat_capacity_numba(RGERG, a0, ar),
        isobaric_heat_capacity_numba(T, D, RGERG, a0, ar),
        isochoric_heat_capacity_numba(RGERG, a0, ar) * 1000 / molar_mass_numba(x),
        isobaric_heat_capacity_numba(T, D, RGERG, a0, ar) * 1000 / molar_mass_numba(x),
        speed_of_sound_numba(T, D, RGERG, a0, ar, x),
        gibbs_energy_numba(RGERG, T, a0, ar),
        joule_thomson_coefficient_numba(T, D, epsilon, RGERG, a0, ar),
        isentropic_exponent_numba(T, D, RGERG, a0, ar, x),
        molar_volume_numba(T, P, x),
        molar_mass_ratio_numba(x),
        specific_gas_constant_numba(x),
    )

    # Compare each value of the refactored results to the original results
    for i, (refactored, original) in enumerate(
        zip(refactored_results, original_results)
    ):
        assert np.isclose(
            refactored, original, rtol=1e-5, atol=1e-8
        ), f"Mismatch in property {i}: {refactored} != {original}"

    print("All properties match with the original PropertiesGERG2008 function.")
