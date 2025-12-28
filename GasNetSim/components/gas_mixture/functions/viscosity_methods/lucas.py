"""
Lucas method implementation for gas mixture viscosity calculation.
Reference: Lucas, K., 1980. Phase Equilibria and Fluid Properties in the Chemical Industry. Dechema, Frankfurt.
"""

import numpy as np
from numba import njit, float64

try:
    from ...viscosity import ViscosityCalculator, MixtureProperties, GAS_PROPERTIES
except ImportError:
    from GasNetSim.components.gas_mixture.viscosity import ViscosityCalculator, MixtureProperties, GAS_PROPERTIES

class LucasViscosityCalculator(ViscosityCalculator):
    """Lucas method implementation for gas mixture viscosity calculation"""

    def __init__(self):
        super().__init__()
        # Register correction factors
        self.register_correction("polarity", self._calculate_polarity_correction)
        self.register_correction("quantum", self._calculate_quantum_correction)
        self.register_correction("mixture", self._calculate_mixture_factor)

        # Register intermediate calculations
        self.register_intermediate("Z1", self._calculate_Z1)
        self.register_intermediate("Z2", self._calculate_Z2)

    @staticmethod
    @njit(float64(float64, float64, float64, float64, float64), cache=True, nogil=True)
    def _calculate_polarity_correction(T, Tc, Pc, Zc, mu):
        """Calculate Lucas method polarity correction factor (FP)

        Parameters:
        -----------
        T : float
            Temperature [K]
        Tc : float
            Critical temperature [K]
        Pc : float
            Critical pressure [Pa]
        Zc : float
            Critical compressibility factor
        mu : float
            Dipole moment [debye]

        Returns:
        --------
        float
            Polarity correction factor
        """
        Pc_bar = Pc / 1e5
        mu_r = 52.46 * mu * mu * Pc_bar / (Tc * Tc)
        Tr = T / Tc

        if mu_r <= 0.022:
            return 1.0
        elif mu_r <= 0.075:
            return 1.0 + 30.55 * (0.292 - Zc) ** 1.72
        else:
            return 1.0 + 30.55 * (0.292 - Zc) ** 1.72 * abs(0.96 + 0.1 * (Tr - 0.7))

    @staticmethod
    @njit(float64(float64, float64, float64), cache=True, nogil=True)
    def _calculate_quantum_correction(T, Tc, M):
        """Calculate Lucas method quantum correction factor (FQ)

        Parameters:
        -----------
        T : float
            Temperature [K]
        Tc : float
            Critical temperature [K]
        M : float
            Molecular weight [g/mol]

        Returns:
        --------
        float
            Quantum correction factor
        """
        Tr = T / Tc

        if abs(M - 4.002602) < 0.0001:  # Helium
            Q = 1.38
        elif abs(M - 2.01588) < 0.0001:  # H2
            Q = 0.76
        else:
            return 1.0

        return 1.22 * Q ** 0.15 * (1 + 0.00385 * (Tr - 12) ** 2) ** (1 / M) * np.sign(Tr - 12)

    @staticmethod
    @njit(float64(float64[:]), cache=True, nogil=True)
    def _calculate_mixture_factor(composition):
        """Calculate Lucas method mixture factor (A)

        Parameters:
        -----------
        composition : numpy.ndarray
            Array of mole fractions

        Returns:
        --------
        float
            Mixture factor
        """
        M = GAS_PROPERTIES[:, 0]

        active_mask = composition > 0
        if not np.any(active_mask):
            return 1.0

        active_M = M[active_mask]
        M_H = np.max(active_M)
        M_L = np.min(active_M)

        heaviest_idx = np.argmax(M * (composition > 0))
        y_H = composition[heaviest_idx]

        if M_H / M_L > 9 and 0.05 <= y_H <= 0.7:
            return 1.0 + 0.01 * (M_H / M_L) ** 0.87
        else:
            return 1.0

    @staticmethod
    @njit(float64(float64, float64, float64), cache=True, nogil=True)
    def _calculate_Z1(Tr, FP_mix, FQ_mix):
        """Calculate Lucas method Z1 factor

        Parameters:
        -----------
        Tr : float
            Reduced temperature
        FP_mix : float
            Mixture polarity correction
        FQ_mix : float
            Mixture quantum correction

        Returns:
        --------
        float
            Z1 factor
        """
        return (0.807 * Tr ** 0.618 - 0.357 * np.exp(-0.449 * Tr) +
                0.340 * np.exp(-4.058 * Tr) + 0.018) * FP_mix * FQ_mix

    @staticmethod
    @njit(float64(float64, float64, float64), cache=True, nogil=True)
    def _calculate_Z2(Tr, Pr, Z1):
        """Calculate Lucas method Z2 factor

        Parameters:
        -----------
        Tr : float
            Reduced temperature
        Pr : float
            Reduced pressure
        Z1 : float
            Z1 factor

        Returns:
        --------
        float
            Z2 factor
        """
        if Tr <= 1.0:
            alpha = 3.262 + 14.98 * Pr ** 5.508
            beta = 1.390 + 5.746 * Pr
            return 0.6 + 0.76 * Pr ** alpha + (6.99 * Pr ** beta - 0.6) * (1 - Tr)
        else:
            a = 1.245e-3 / Tr * np.exp(5.1726 * Tr ** (-0.3286))
            b = a * (1.6553 * Tr - 1.2723)
            c = 0.4489 / Tr * np.exp(3.0578 * Tr ** (-37.7332))
            d = 1.7368 / Tr * np.exp(2.2310 * Tr ** (-7.6351))
            e = 1.3088
            f = 0.9425 * np.exp(-0.1853 * Tr ** 0.4489)
            return Z1 * (1 + (a * Pr ** e) / (b * Pr ** f + (1 / (a + c * Pr ** d))))

    def calculate_viscosity(self, props: MixtureProperties) -> float:
        """
        Calculate mixture viscosity using Lucas method

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

        # Calculate correction factors
        FP_values = np.zeros_like(composition)
        FQ_values = np.zeros_like(composition)

        for i in range(len(composition)):
            FP_values[i] = self.get_correction(
                "polarity",
                props.T, GAS_PROPERTIES[i, 1], GAS_PROPERTIES[i, 2],
                GAS_PROPERTIES[i, 6], GAS_PROPERTIES[i, 5]
            )
            FQ_values[i] = self.get_correction(
                "quantum",
                props.T, GAS_PROPERTIES[i, 1], GAS_PROPERTIES[i, 0]
            )

        A = self.get_correction("mixture", composition)

        FP_mix = np.sum(composition * FP_values)
        FQ_mix = np.sum(composition * FQ_values) * A

        # Calculate intermediate values
        Z1 = self.get_intermediate("Z1", props.Tr, FP_mix, FQ_mix)
        Z2 = self.get_intermediate("Z2", props.Tr, props.Pr, Z1)

        # Calculate final corrections
        Y = Z2 / Z1
        FP = (1 + (FP_mix - 1) * Y ** (-3)) / FP_mix
        FQ = (1 + (FQ_mix - 1) * (Y ** (-1) - 0.007 * np.log(Y) ** 4)) / FQ_mix

        # Calculate final viscosity
        Pc_bar = props.Pc_mix / 1e5
        xi = 0.176 * (props.Tc_mix / (props.M_mix ** 3 * Pc_bar ** 4)) ** (1 / 6)
        eta = Z2 * FP * FQ / xi

        return eta / 1e7  # Convert to Pa*s


if __name__ == "__main__":
    from GasNetSim.components.gas_mixture.viscosity import ViscosityCalculator, MixtureProperties, GAS_PROPERTIES
    # Example mixtures
    print("\nLucas Method Viscosity Calculator Test")
    print("-" * 50)

    # Test conditions
    test_conditions = [
        {
            "name": "Pure methane",
            "T": 273.15,  # K
            "P": 1e6,  # Pa (10 bar)
            "composition": np.zeros(21),  # Initialize with zeros
        },
        {
            "name": "Methane-nitrogen mixture",
            "T": 273.15,  # K
            "P": 1e7,  # Pa (100 bar)
            "composition": np.zeros(21),
        },
        {
            "name": "Natural gas mixture",
            "T": 350.0,  # K
            "P": 10e6,  # Pa (100 bar)
            "composition": np.zeros(21),
        }
    ]

    # Set compositions
    test_conditions[0]["composition"][0] = 1.0  # Pure methane

    test_conditions[1]["composition"][0] = 0.95  # Methane
    test_conditions[1]["composition"][1] = 0.05  # Nitrogen

    test_conditions[2]["composition"][0] = 0.90  # Methane
    test_conditions[2]["composition"][1] = 0.05  # Nitrogen
    test_conditions[2]["composition"][2] = 0.03  # CO2
    test_conditions[2]["composition"][3] = 0.02  # Ethane

    # Create calculator
    calculator = LucasViscosityCalculator()

    # Test each condition
    for condition in test_conditions:
        print(f"\nTesting: {condition['name']}")
        print(f"T = {condition['T']:.2f} K")
        print(f"P = {condition['P'] / 1e6:.2f} MPa")

        # Create properties object
        props = MixtureProperties(condition['T'], condition['P'], condition['composition'])

        # Calculate viscosity
        try:
            viscosity = calculator.calculate_viscosity(props)
            print(f"Viscosity = {viscosity * 1e6:.3f} μPa·s")
        except Exception as e:
            print(f"Error: {str(e)}")

        print("-" * 50)