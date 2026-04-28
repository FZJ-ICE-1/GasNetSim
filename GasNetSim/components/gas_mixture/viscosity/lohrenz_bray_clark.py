#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
import numpy as np

from .core import (
    VISCOSITY_COMPONENT_PROPERTIES,
    ViscosityCalculator,
    ViscosityState,
)


class LBCViscosityCalculator(ViscosityCalculator):
    """Lohrenz-Bray-Clark (LBC) method implementation for gas mixture viscosity calculation."""

    def calculate_viscosity(self, props: ViscosityState) -> float:
        if props.density is None:
            raise ValueError("Density is required for LBC viscosity calculation")

        component_viscosities = self._calculate_component_viscosities(props)
        eta_mix = self._calculate_mixture_viscosity(props, component_viscosities)
        rho_r = self._calculate_reduced_density(props)
        delta = 1.023 + 0.23364 * rho_r + 0.58533 * rho_r**2 - 0.40758 * rho_r**3 + 0.093324 * rho_r**4
        xi = self._calculate_xi_parameter(props)
        return (eta_mix + xi * (delta**4 - 1)) * 1e-3

    def _calculate_component_viscosities(self, props: ViscosityState) -> np.ndarray:
        u_eta = 0.0001
        u_M = 1.0
        u_T = 1.0
        u_P = 0.101325

        n_components = len(props.composition)
        component_viscosities = np.zeros(n_components)

        for i in range(n_components):
            if props.composition[i] > 0:
                molecular_weight = VISCOSITY_COMPONENT_PROPERTIES[i, 0]
                critical_temperature = VISCOSITY_COMPONENT_PROPERTIES[i, 1]
                critical_pressure = VISCOSITY_COMPONENT_PROPERTIES[i, 2] / 1e6
                reduced_temperature = props.T / critical_temperature

                if reduced_temperature <= 1.5:
                    alpha = 3.4 * reduced_temperature**0.94
                else:
                    alpha = 1.778 * (4.58 * reduced_temperature - 1.67) ** 0.625

                component_viscosities[i] = (
                    u_eta
                    * (molecular_weight / u_M) ** 0.5
                    * (critical_temperature / u_T) ** (-1 / 6)
                    * (critical_pressure / u_P) ** (2 / 3)
                    * alpha
                )

        return component_viscosities

    def _calculate_mixture_viscosity(self, props: ViscosityState, component_viscosities: np.ndarray) -> float:
        numerator = 0.0
        denominator = 0.0

        for i in range(len(props.composition)):
            if props.composition[i] > 0:
                molecular_weight = VISCOSITY_COMPONENT_PROPERTIES[i, 0]
                numerator += props.composition[i] * component_viscosities[i] * np.sqrt(molecular_weight)
                denominator += props.composition[i] * np.sqrt(molecular_weight)

        return numerator / denominator

    def _calculate_reduced_density(self, props: ViscosityState) -> float:
        molar_density = props.density / (props.M_mix / 1000)
        critical_volume_mix = 0.0

        for i in range(len(props.composition)):
            if props.composition[i] > 0:
                critical_volume_mix += props.composition[i] / VISCOSITY_COMPONENT_PROPERTIES[i, 3]

        if critical_volume_mix > 0:
            critical_volume_mix = 1.0 / critical_volume_mix

        return critical_volume_mix * molar_density

    def _calculate_xi_parameter(self, props: ViscosityState) -> float:
        u_eta = 0.0001
        u_M = 1.0
        u_T = 1.0
        u_P = 0.101325
        return (
            u_eta
            * (props.M_mix / u_M) ** 0.5
            * (props.Tc_mix / u_T) ** (-1 / 6)
            * (props.Pc_mix / 1e6 / u_P) ** (2 / 3)
        )
