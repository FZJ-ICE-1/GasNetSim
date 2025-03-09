"""
Lohrenz-Bray-Clark method implementation for gas mixture viscosity calculation.
Reference: ISO 20765-5:2022 - Natural gas — Calculation of thermodynamic properties — Part 5:
Calculation of viscosity, Joule-Thomson coefficient, and isentropic exponent
"""

import numpy as np
from numba import njit, float64
try:
    from ...viscosity import ViscosityCalculator, MixtureProperties, GAS_PROPERTIES
except ImportError:
    from GasNetSim.components.gas_mixture.viscosity import ViscosityCalculator, MixtureProperties, GAS_PROPERTIES

class LBCViscosityCalculator(ViscosityCalculator):
    """Lohrenz-Bray-Clark (LBC) method implementation for gas mixture viscosity calculation"""

    def calculate_viscosity(self, props: MixtureProperties) -> float:
        """
        Calculate mixture viscosity using Lohrenz-Bray-Clark method

        Parameters:
        -----------
        props : MixtureProperties
            Container with mixture properties

        Returns:
        --------
        float
            Viscosity in Pa*s
        """
        if props.density is None:
            raise ValueError("Density is required for LBC viscosity calculation")

        # Get component properties
        composition = props.composition
        T = props.T

        # Calculate pure component viscosities
        component_viscosities = self._calculate_component_viscosities(props)

        # Calculate generalized mixture viscosity (equation 10)
        eta_mix = self._calculate_mixture_viscosity(props, component_viscosities)

        # Calculate density parameter
        rho_r = self._calculate_reduced_density(props)

        # Calculate delta correction (equation 13)
        delta = 1.023 + 0.23364 * rho_r + 0.58533 * rho_r**2 - 0.40758 * rho_r**3 + 0.093324 * rho_r**4

        # Calculate xi parameter (equation 11)
        xi = self._calculate_xi_parameter(props)

        # Final LBC formula (equation 9)
        # Converting mPa·s to Pa·s by multiplying by 1e-3
        return (eta_mix + xi * (delta**4 - 1)) * 1e-3

    def _calculate_component_viscosities(self, props: MixtureProperties) -> np.ndarray:
        """
        Calculate viscosities of pure components

        Parameters:
        -----------
        props : MixtureProperties
            Container with mixture properties

        Returns:
        --------
        numpy.ndarray
            Array of component viscosities in mPa·s
        """
        # Constants from ISO 20765-5:2022 (equation 12)
        u_eta = 0.0001  # mPa·s
        u_M = 1.0       # g/mol
        u_T = 1.0       # K
        u_P = 0.101325  # MPa

        # Initialize array for component viscosities
        n_components = len(props.composition)
        component_viscosities = np.zeros(n_components)

        # Only calculate for components with non-zero composition
        for i in range(n_components):
            if props.composition[i] > 0:
                # Extract component properties
                M_i = GAS_PROPERTIES[i, 0]     # Molecular weight [g/mol]
                Tc_i = GAS_PROPERTIES[i, 1]    # Critical temperature [K]
                Pc_i = GAS_PROPERTIES[i, 2]/1e6 # Critical pressure [MPa]

                # Calculate reduced temperature
                Tr_i = props.T / Tc_i

                # Calculate alpha parameter based on reduced temperature (equations 16, 17)
                if Tr_i <= 1.5:
                    alpha_i = 3.4 * Tr_i**0.94
                else:
                    alpha_i = 1.778 * (4.58 * Tr_i - 1.67)**0.625

                # Component viscosity calculation (equation 15)
                component_viscosities[i] = u_eta * (M_i/u_M)**(1/2) * (Tc_i/u_T)**(-1/6) * (Pc_i/u_P)**(2/3) * alpha_i

        return component_viscosities

    def _calculate_mixture_viscosity(self, props: MixtureProperties, component_viscosities: np.ndarray) -> float:
        """
        Calculate the generalized mixture viscosity

        Parameters:
        -----------
        props : MixtureProperties
            Container with mixture properties
        component_viscosities : numpy.ndarray
            Array of component viscosities

        Returns:
        --------
        float
            Generalized mixture viscosity in mPa·s
        """
        # Initialize variables for summation
        numerator = 0.0
        denominator = 0.0

        # Apply mixing rule (equation 10)
        for i in range(len(props.composition)):
            if props.composition[i] > 0:
                M_i = GAS_PROPERTIES[i, 0]  # Molecular weight
                numerator += props.composition[i] * component_viscosities[i] * np.sqrt(M_i)
                denominator += props.composition[i] * np.sqrt(M_i)

        # Return mixture viscosity
        return numerator / denominator

    def _calculate_reduced_density(self, props: MixtureProperties) -> float:
        """
        Calculate the reduced density parameter

        Parameters:
        -----------
        props : MixtureProperties
            Container with mixture properties

        Returns:
        --------
        float
            Reduced density
        """
        # Calculate molar density (mol/m³)
        molar_density = props.density / (props.M_mix / 1000)  # kg/m³ / (g/mol / 1000) = mol/m³

        # Calculate critical volume (m³/mol)
        Vc_mix = 0.0
        for i in range(len(props.composition)):
            if props.composition[i] > 0:
                Vc_i = GAS_PROPERTIES[i, 3]  # Critical volume [m³/mol]
                Vc_mix += props.composition[i] / GAS_PROPERTIES[i, 3]

        if Vc_mix > 0:
            Vc_mix = 1.0 / Vc_mix

        # Calculate reduced density (equation 14)
        return Vc_mix * molar_density

    def _calculate_xi_parameter(self, props: MixtureProperties) -> float:
        """
        Calculate the xi parameter for density correction

        Parameters:
        -----------
        props : MixtureProperties
            Container with mixture properties

        Returns:
        --------
        float
            Xi parameter in mPa·s
        """
        # Constants from the standard (equation 12)
        u_eta = 0.0001  # mPa·s
        u_M = 1.0       # g/mol
        u_T = 1.0       # K
        u_P = 0.101325  # MPa

        # Xi calculation (equation 11)
        return u_eta * (props.M_mix/u_M)**(1/2) * (props.Tc_mix/u_T)**(-1/6) * (props.Pc_mix/1e6/u_P)**(2/3)


if __name__ == "__main__":
    # Example mixtures
    print("\nLohrenz-Bray-Clark Method Viscosity Calculator Test")
    print("-" * 50)

    # Test conditions
    test_conditions = [
        {
            "name": "Pure methane",
            "T": 273.15,  # K
            "P": 1e5,     # Pa (1 bar)
            "composition": np.zeros(21),  # Initialize with zeros
            "density": 0.717  # kg/m³
        },
        {
            "name": "Methane-nitrogen mixture",
            "T": 300.0,  # K
            "P": 5e6,    # Pa (50 bar)
            "composition": np.zeros(21),
            "density": 35.0  # kg/m³
        },
        {
            "name": "Natural gas mixture",
            "T": 293.15,  # K (20°C)
            "P": 7e6,     # Pa (70 bar)
            "composition": np.zeros(21),
            "density": 55.0  # kg/m³
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
    calculator = LBCViscosityCalculator()

    # Test each condition
    for condition in test_conditions:
        print(f"\nTesting: {condition['name']}")
        print(f"T = {condition['T']:.2f} K")
        print(f"P = {condition['P'] / 1e5:.2f} bar")
        print(f"Density = {condition['density']:.3f} kg/m³")

        # Create properties object
        props = MixtureProperties(condition['T'], condition['P'], condition['composition'])
        props.density = condition['density']

        # Calculate viscosity
        try:
            viscosity = calculator.calculate_viscosity(props)
            print(f"Viscosity = {viscosity * 1e6:.3f} μPa·s")
        except Exception as e:
            print(f"Error: {str(e)}")

        print("-" * 50)