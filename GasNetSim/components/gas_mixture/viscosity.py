"""
Gas mixture viscosity calculation module.
Implements various methods for calculating gas mixture viscosity:
- Lucas method
- Chapman-Enskog method
- TRAPP method
- Carr method
- Chung method
"""
import numpy as np
from enum import Enum
from numba import njit, float64

from .functions.viscosity_methods.base_method import ViscosityCalculator, MixtureProperties, GAS_PROPERTIES, R_GAS
from .functions.viscosity_methods.herning_zipperer import HerningZippererCalculator


class ViscosityMethod(Enum):
    """Available methods for viscosity calculation"""
    LUCAS = "lucas"
    CHAPMAN_ENSKOG = "chapman_enskog"
    TRAPP = "trapp"
    HERNING_ZIPPERER = "herning_zipperer"
    CHUNG = "chung"

class ViscosityCalculatorFactory:
    """Factory for creating viscosity calculators"""
    _calculators = {}

    @classmethod
    def register_calculator(cls, method: ViscosityMethod, calculator_class):
        """Register a calculator class for a method"""
        cls._calculators[method] = calculator_class

    @classmethod
    def get_calculator(cls, method: ViscosityMethod) -> ViscosityCalculator:
        """Get appropriate calculator for the method"""
        if method not in cls._calculators:
            raise ValueError(f"Unsupported viscosity calculation method: {method}")
        return cls._calculators[method]()


def calculate_viscosity(T: float, P: float, composition, method=ViscosityMethod.LUCAS, density=None):
    """
    Calculate gas mixture viscosity using the specified method

    Parameters:
    -----------
    T : float
        Temperature in Kelvin
    P : float
        Pressure in Pascal
    composition : numpy.ndarray
        Array of mole fractions for each component
    method : ViscosityMethod
        Method to use for viscosity calculation
    density : float, optional
        Mixture density in kg/m3 (required for some methods)

    Returns:
    --------
    float
        Viscosity in Pa*s
    """
    props = MixtureProperties(T, P, composition)
    if density is not None:
        props.density = density

    calculator = ViscosityCalculatorFactory.get_calculator(method)
    return calculator.calculate_viscosity(props)


# Import and register calculators
from .functions.viscosity_methods.lucas import LucasViscosityCalculator
from .functions.viscosity_methods.herning_zipperer import HerningZippererCalculator
# from ..viscosity_methods.chapman_enskog import ChapmanEnskogCalculator
# from ..viscosity_methods.trapp import TrappCalculator
# from ..viscosity_methods.chung import ChungCalculator

ViscosityCalculatorFactory.register_calculator(ViscosityMethod.LUCAS, LucasViscosityCalculator)
# ViscosityCalculatorFactory.register_calculator(ViscosityMethod.CHAPMAN_ENSKOG, ChapmanEnskogCalculator)
# ViscosityCalculatorFactory.register_calculator(ViscosityMethod.TRAPP, TrappCalculator)
ViscosityCalculatorFactory.register_calculator(ViscosityMethod.HERNING_ZIPPERER, HerningZippererCalculator)
# ViscosityCalculatorFactory.register_calculator(ViscosityMethod.CHUNG, ChungCalculator)

# Define what's available when importing *
__all__ = [
    'calculate_viscosity',
    'ViscosityMethod',
    'MixtureProperties',
    'ViscosityCalculator',
    'ViscosityCalculatorFactory',
    'LucasViscosityCalculator',
    # 'ChapmanEnskogCalculator',
    # 'TrappCalculator',
    'HerningZippererCalculator',
    # 'ChungCalculator'
]