#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
import numpy as np
from numba import float64, njit

from .core import (
    VISCOSITY_COMPONENT_PROPERTIES,
    ViscosityCalculator,
    ViscosityState,
)


class LucasViscosityCalculator(ViscosityCalculator):
    """Lucas method implementation for gas mixture viscosity calculation."""

    def __init__(self):
        super().__init__()
        self.register_correction("polarity", self._calculate_polarity_correction)
        self.register_correction("quantum", self._calculate_quantum_correction)
        self.register_correction("mixture", self._calculate_mixture_factor)
        self.register_intermediate("Z1", self._calculate_Z1)
        self.register_intermediate("Z2", self._calculate_Z2)

    @staticmethod
    @njit(float64(float64, float64, float64, float64, float64), cache=True, nogil=True)
    def _calculate_polarity_correction(T, Tc, Pc, Zc, mu):
        pressure_bar = Pc / 1e5
        mu_r = 52.46 * mu * mu * pressure_bar / (Tc * Tc)
        Tr = T / Tc

        if mu_r <= 0.022:
            return 1.0
        if mu_r <= 0.075:
            return 1.0 + 30.55 * (0.292 - Zc) ** 1.72
        return 1.0 + 30.55 * (0.292 - Zc) ** 1.72 * abs(0.96 + 0.1 * (Tr - 0.7))

    @staticmethod
    @njit(float64(float64, float64, float64), cache=True, nogil=True)
    def _calculate_quantum_correction(T, Tc, M):
        Tr = T / Tc

        if abs(M - 4.002602) < 0.0001:
            Q = 1.38
        elif abs(M - 2.01588) < 0.0001:
            Q = 0.76
        else:
            return 1.0

        return 1.22 * Q ** 0.15 * (1 + 0.00385 * (Tr - 12) ** 2) ** (1 / M) * np.sign(Tr - 12)

    @staticmethod
    @njit(float64(float64[:]), cache=True, nogil=True)
    def _calculate_mixture_factor(composition):
        molecular_weights = VISCOSITY_COMPONENT_PROPERTIES[:, 0]
        active_mask = composition > 0
        if not np.any(active_mask):
            return 1.0

        active_weights = molecular_weights[active_mask]
        M_H = np.max(active_weights)
        M_L = np.min(active_weights)
        heaviest_idx = np.argmax(molecular_weights * (composition > 0))
        y_H = composition[heaviest_idx]

        if M_H / M_L > 9 and 0.05 <= y_H <= 0.7:
            return 1.0 + 0.01 * (M_H / M_L) ** 0.87
        return 1.0

    @staticmethod
    @njit(float64(float64, float64, float64), cache=True, nogil=True)
    def _calculate_Z1(Tr, FP_mix, FQ_mix):
        return (
            0.807 * Tr ** 0.618
            - 0.357 * np.exp(-0.449 * Tr)
            + 0.340 * np.exp(-4.058 * Tr)
            + 0.018
        ) * FP_mix * FQ_mix

    @staticmethod
    @njit(float64(float64, float64, float64), cache=True, nogil=True)
    def _calculate_Z2(Tr, Pr, Z1):
        if Tr <= 1.0:
            alpha = 3.262 + 14.98 * Pr ** 5.508
            beta = 1.390 + 5.746 * Pr
            return 0.6 + 0.76 * Pr ** alpha + (6.99 * Pr ** beta - 0.6) * (1 - Tr)

        a = 1.245e-3 / Tr * np.exp(5.1726 * Tr ** (-0.3286))
        b = a * (1.6553 * Tr - 1.2723)
        c = 0.4489 / Tr * np.exp(3.0578 * Tr ** (-37.7332))
        d = 1.7368 / Tr * np.exp(2.2310 * Tr ** (-7.6351))
        e = 1.3088
        f = 0.9425 * np.exp(-0.1853 * Tr ** 0.4489)
        return Z1 * (1 + (a * Pr ** e) / (b * Pr ** f + (1 / (a + c * Pr ** d))))

    def calculate_viscosity(self, props: ViscosityState) -> float:
        composition = props.composition
        FP_values = np.zeros_like(composition)
        FQ_values = np.zeros_like(composition)

        for i in range(len(composition)):
            FP_values[i] = self.get_correction(
                "polarity",
                props.T,
                VISCOSITY_COMPONENT_PROPERTIES[i, 1],
                VISCOSITY_COMPONENT_PROPERTIES[i, 2],
                VISCOSITY_COMPONENT_PROPERTIES[i, 6],
                VISCOSITY_COMPONENT_PROPERTIES[i, 5],
            )
            FQ_values[i] = self.get_correction(
                "quantum",
                props.T,
                VISCOSITY_COMPONENT_PROPERTIES[i, 1],
                VISCOSITY_COMPONENT_PROPERTIES[i, 0],
            )

        mixture_factor = self.get_correction("mixture", composition)
        FP_mix = np.sum(composition * FP_values)
        FQ_mix = np.sum(composition * FQ_values) * mixture_factor
        Z1 = self.get_intermediate("Z1", props.Tr, FP_mix, FQ_mix)
        Z2 = self.get_intermediate("Z2", props.Tr, props.Pr, Z1)

        Y = Z2 / Z1
        FP = (1 + (FP_mix - 1) * Y ** (-3)) / FP_mix
        FQ = (1 + (FQ_mix - 1) * (Y ** (-1) - 0.007 * np.log(Y) ** 4)) / FQ_mix

        pressure_bar = props.Pc_mix / 1e5
        xi = 0.176 * (props.Tc_mix / (props.M_mix ** 3 * pressure_bar ** 4)) ** (1 / 6)
        eta = Z2 * FP * FQ / xi
        return eta / 1e7
