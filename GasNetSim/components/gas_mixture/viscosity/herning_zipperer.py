#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
import numpy as np

from .core import (
    VISCOSITY_COMPONENT_PROPERTIES,
    ViscosityCalculator,
    ViscosityState,
)


PURE_GAS_VISCOSITY = np.array([
    10.2, 16.58, 13.83, 8.6, 7.5, 6.9, 6.9, 6.2, 6.2, 5.9,
    4.99, 5.5, 5.5, 5.5, 8.44, 19.23, 16.58, 8.75, 11.68, 18.5, 20.93,
], dtype=np.float64)


class HerningZippererCalculator(ViscosityCalculator):
    """Herning-Zipperer method implementation for gas mixture viscosity calculation."""

    def calculate_viscosity(self, props: ViscosityState) -> float:
        composition = props.composition
        numerator = 0.0
        denominator = 0.0

        for i in range(len(composition)):
            if composition[i] > 0 and i < len(PURE_GAS_VISCOSITY):
                molecular_weight = VISCOSITY_COMPONENT_PROPERTIES[i, 0]
                eta_i = PURE_GAS_VISCOSITY[i] * 1e-6
                numerator += composition[i] * eta_i * np.sqrt(molecular_weight)
                denominator += composition[i] * np.sqrt(molecular_weight)

        return numerator / denominator
