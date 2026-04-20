import numpy as np
from numba import njit, float64
from dataclasses import dataclass
from typing import Dict, Callable, Optional

# Constants
R_GAS = 8.314  # Gas constant in J/(mol*K)

# Gas properties array [M, Tc, Pc, Vc, omega, mu, Zc]
GAS_PROPERTIES = np.array([
    [16.04246, 190.564, 4599200.0, 9.86278109912e-05, 0.01142, 0.0, 0.28629030721213733],  # methane
    [28.0134, 126.192, 3395800.0, 8.94142472662e-05, 0.0372, 0.0, 0.28938953385060845],  # nitrogen
    [44.0095, 304.1282, 7377300.0, 9.41184770731e-05, 0.22394, 0.0, 0.27458794013625015],  # carbon dioxide
    [30.06904, 305.322, 4872200.0, 0.000145838781642, 0.0995, 0.0, 0.2799019031095579],  # ethane
    [44.09562, 369.89, 4251200.0, 0.0002, 0.1521, 0.08, 0.2764615619549796],  # propane
    [58.1222, 407.81, 3629000.0, 0.000257748115318, 0.184, 0.13, 0.2758610662463119],  # isobutane
    [58.1222, 425.125, 3796000.0, 0.000254921929824, 0.201, 0.0, 0.27376792941858136],  # n-butane
    [72.14878, 460.35, 3378000.0, 0.000305716906145, 0.2274, 0.13, 0.26980920889878157],  # isopentane
    [72.14878, 469.7, 3367500.0, 0.000311526479751, 0.251, 0.0, 0.268625864860163],  # n-pentane
    [86.17536, 507.82, 3044100.0, 0.000369549150037, 0.3, 0.0, 0.266432461189194],  # n-hexane
    [100.20194, 540.2, 2735730.0, 0.000429184549356, 0.349, 0.0, 0.2614138221425429],  # n-heptane
    [114.22852, 568.74, 2483590.0, 0.000492368291482, 0.398, 0.0, 0.2585961295289495],  # n-octane
    [128.2551, 594.55, 2281000.0, 0.000552486187845, 0.4433, 0.0, 0.2549318760066063],  # n-nonane
    [142.28168, 617.7, 2103000.0, 0.000609756097561, 0.4884, 0.0, 0.2496799324943269],  # n-decane
    [2.01588, 33.145, 1296400.0, 6.44828475625e-05, -0.219, 0.0, 0.3033409353716562],  # hydrogen
    [31.9988, 154.581, 5043000.0, 7.33675715334e-05, 0.0222, 0.0, 0.28787424687871216],  # oxygen
    [28.0101, 132.86, 3494000.0, 9.21658986175e-05, 0.0497, 0.11, 0.2915175660594411],  # carbon monoxide
    [18.01528, 647.096, 22064000.0, 5.59480372671e-05, 0.3443, 1.85, 0.22943845208106295],  # water
    [34.08088, 373.1, 9000000.0, 9.81354268891e-05, 0.1005, 0.97, 0.2847140448825459],  # hydrogen sulfide
    [4.002602, 5.1953, 228320.0, 5.75251528731e-05, -0.3836, 0.0, 0.304058340910078],  # helium
    [39.948, 150.687, 4863000.0, 7.45855116234e-05, -0.00219, 0.0, 0.2895001352576394],  # argon
])


@dataclass
class MixtureProperties:
    """Container for mixture properties"""
    T: float  # Temperature [K]
    P: float  # Pressure [Pa]
    composition: np.ndarray  # Mole fractions
    density: Optional[float] = None  # Density [kg/m3]
    Tc_mix: float = 0.0  # Critical temperature [K]
    Pc_mix: float = 0.0  # Critical pressure [Pa]
    M_mix: float = 0.0  # Molecular weight [g/mol]
    Tr: float = 0.0  # Reduced temperature
    Pr: float = 0.0  # Reduced pressure
    Zc_mix: float = 0.0  # Critical compressibility factor
    Vc_mix: float = 0.0  # Critical volume [m3/mol]
    omega_mix: float = 0.0  # Acentric factor
    mu_mix: float = 0.0  # Dipole moment [debye]

    def __post_init__(self):
        """Calculate mixture properties after initialization"""
        props = calculate_mixture_critical_point(self.composition)
        self.Tc_mix, self.Pc_mix, self.M_mix = props[0], props[1], props[2]
        self.Vc_mix = props[3]
        self.omega_mix = props[4]
        self.mu_mix = props[5]
        self.Zc_mix = props[6]
        self.Tr = self.T / self.Tc_mix
        self.Pr = self.P / self.Pc_mix


class ViscosityCalculator:
    """Base class for viscosity calculations"""

    def __init__(self):
        self._correction_factors: Dict[str, Callable] = {}
        self._intermediate_calcs: Dict[str, Callable] = {}

    def register_correction(self, name: str, func: Callable):
        """Register a correction factor calculation"""
        self._correction_factors[name] = func

    def register_intermediate(self, name: str, func: Callable):
        """Register an intermediate calculation"""
        self._intermediate_calcs[name] = func

    def get_correction(self, name: str, *args, **kwargs):
        """Get correction factor value"""
        if name not in self._correction_factors:
            raise KeyError(f"Correction factor {name} not registered")
        return self._correction_factors[name](*args, **kwargs)

    def get_intermediate(self, name: str, *args, **kwargs):
        """Get intermediate calculation value"""
        if name not in self._intermediate_calcs:
            raise KeyError(f"Intermediate calculation {name} not registered")
        return self._intermediate_calcs[name](*args, **kwargs)

    def calculate_viscosity(self, props: MixtureProperties) -> float:
        """
        Calculate mixture viscosity

        Parameters:
        -----------
        props : MixtureProperties
            Container with mixture properties

        Returns:
        --------
        float
            Viscosity in Pa*s
        """
        raise NotImplementedError("Subclasses must implement calculate_viscosity")


@njit(float64[:](float64[:]), cache=True, nogil=True)
def calculate_mixture_critical_point(composition):
    """Calculate mixture critical point properties using linear mixing rules

    Parameters:
    -----------
    composition : numpy.ndarray
        Array of mole fractions for each component

    Returns:
    --------
    numpy.ndarray
        Array of [Tc_mix, Pc_mix, M_mix, Vc_mix, omega_mix, mu_mix, Zc_mix]
    """
    M = GAS_PROPERTIES[:, 0]
    Tc = GAS_PROPERTIES[:, 1]
    Vc = GAS_PROPERTIES[:, 3]
    omega = GAS_PROPERTIES[:, 4]  # Acentric factor
    mu = GAS_PROPERTIES[:, 5]  # Dipole moment
    Zc = GAS_PROPERTIES[:, 6]

    M_mix = np.sum(composition * M)
    Tc_mix = np.sum(composition * Tc)
    Vc_mix = np.sum(composition * Vc)
    Zc_mix = np.sum(composition * Zc)
    omega_mix = np.sum(composition * omega)
    mu_mix = np.sum(composition * mu)

    Pc_mix = Zc_mix * R_GAS * Tc_mix / Vc_mix

    return np.array([Tc_mix, Pc_mix, M_mix, Vc_mix, omega_mix, mu_mix, Zc_mix])