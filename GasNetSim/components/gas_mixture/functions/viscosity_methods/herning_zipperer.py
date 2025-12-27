"""
Herning-Zipperer method implementation for gas mixture viscosity calculation.
Reference: F. Herning, L. Zipperer, Gas-und Wasserfach 79 (1936) 49–54.
"""

import numpy as np
from numba import njit, float64
try:
    from ...viscosity import ViscosityCalculator, MixtureProperties, GAS_PROPERTIES
except ImportError:
    from GasNetSim.components.gas_mixture.viscosity import ViscosityCalculator, MixtureProperties, GAS_PROPERTIES

PURE_GAS_VISCOSITY = np.array([
    10.2,   # CH4 (Methane)
    16.58,  # N2 (Nitrogen)
    13.83,  # CO2 (Carbon dioxide)
    8.6,    # C2H6 (Ethane)
    7.5,    # C3H8 (Propane)
    6.9,    # i-C4H10 (Isobutane)
    6.9,    # n-C4H10 (n-Butane)
    6.2,    # i-C5H12 (Isopentane)
    6.2,    # n-C5H12 (n-Pentane)
    5.9,    # n-C6H14 (n-Hexane)
    4.99,    # n-C7H16 (n-Heptane)
    5.5,   # n-C8H18 (n-Octane)
    5.5,    # n-C9H20 (n-Nonane)
    5.5,    # n-C10H22 (n-Decane)
    8.44,   # H2 (Hydrogen)
    19.23,  # O2 (Oxygen)
    16.58,  # CO (Carbon monoxide)
    8.75,   # H2O (Water)
    11.68,  # H2S (Hydrogen sulfide)
    18.5,   # He (Helium)
    20.93,  # Ar (Argon)
], dtype=np.float64)

class HerningZippererCalculator(ViscosityCalculator):
    """Herning-Zipperer method implementation for gas mixture viscosity calculation"""

    def calculate_viscosity(self, props: MixtureProperties) -> float:
        """
        Calculate mixture viscosity using Herning-Zipperer method

        Parameters:
        -----------
        props : MixtureProperties
            Container with mixture properties

        Returns:
        --------
        float
            Viscosity in Pa*s
        """
        composition = props.composition

        # Calculate pure component viscosities
        numerator = 0.0
        denominator = 0.0

        for i in range(len(composition)):
            if composition[i] > 0 and i < len(PURE_GAS_VISCOSITY):
                # Get molecular weight and tabulated viscosity
                M_i = GAS_PROPERTIES[i, 0]
                eta_i = PURE_GAS_VISCOSITY[i] * 1e-6  # Convert μPa·s to Pa·s

                # Apply Herning-Zipperer mixing rule
                numerator += composition[i] * eta_i * np.sqrt(M_i)
                denominator += composition[i] * np.sqrt(M_i)

        # Return mixture viscosity
        return numerator / denominator


if __name__ == "__main__":
    # Example mixtures
    print("\nHerning-Zipperer Method Viscosity Calculator Test")
    print("-" * 50)

    # Test conditions
    test_conditions = [
        {
            "name": "Pure methane",
            "T": 273.15,  # K
            "P": 1e5,  # Pa (1 bar)
            "composition": np.zeros(21),  # Initialize with zeros
        },
        {
            "name": "Methane-nitrogen mixture",
            "T": 300.0,  # K
            "P": 1e5,  # Pa (1 bar)
            "composition": np.zeros(21),
        },
        {
            "name": "Natural gas mixture",
            "T": 350.0,  # K
            "P": 1e5,  # Pa (1 bar)
            "composition": np.zeros(21),
        }
    ]

    # Set compositions
    test_conditions[0]["composition"][0] = 1.0  # Pure methane

    test_conditions[1]["composition"][0] = 0.85  # Methane
    test_conditions[1]["composition"][1] = 0.15  # Nitrogen

    test_conditions[2]["composition"][0] = 0.90  # Methane
    test_conditions[2]["composition"][1] = 0.05  # Nitrogen
    test_conditions[2]["composition"][2] = 0.03  # CO2
    test_conditions[2]["composition"][3] = 0.02  # Ethane

    # Create calculator
    calculator = HerningZippererCalculator()

    # Test each condition
    for condition in test_conditions:
        print(f"\nTesting: {condition['name']}")
        print(f"T = {condition['T']:.2f} K")
        print(f"P = {condition['P'] / 1e5:.2f} bar")

        # Create properties object
        props = MixtureProperties(condition['T'], condition['P'], condition['composition'])

        # Calculate viscosity
        try:
            viscosity = calculator.calculate_viscosity(props)
            print(f"Viscosity = {viscosity * 1e6:.3f} μPa·s")
        except Exception as e:
            print(f"Error: {str(e)}")

        print("-" * 50)