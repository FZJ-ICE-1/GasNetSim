"""
Dynamic Pipeline - subclass of GasNetSim Pipeline with Pi-model properties.

Author: Yetkin Civan Serin
"""

import numpy as np
import math
from GasNetSim.components.pipeline import Pipeline

class DynamicPipeline(Pipeline):
    """
    Pipeline subclass for dynamic Pi-model simulation.
    
    Inherits from the Pipeline Class and adds:
    - Pi-model property aliases (i, j, L, D, A, f, theta)
    - Tilt angle computation from node altitudes
    """
    def __init__(
        self,
        pipeline_index: int,
        inlet,
        outlet,
        diameter,
        length,
        efficiency=0.85,
        roughness=0.000015,
        ambient_temp=288.15,
        ambient_pressure=101325.0,
        heat_transfer_coefficient=3.69,
        valve=0,
        friction_factor_method="constant",
        conversion_factor=1.0,
        constant_friction_factor=None,
    ):
        # Parent constructor
        super().__init__(
            pipeline_index=pipeline_index,
            inlet=inlet,
            outlet=outlet,
            diameter=diameter,
            length=length,
            efficiency=efficiency,
            roughness=roughness,
            ambient_temp=ambient_temp,
            ambient_pressure=ambient_pressure,
            heat_transfer_coefficient=heat_transfer_coefficient,
            valve=valve,
            friction_factor_method=friction_factor_method,
            conversion_factor=conversion_factor,
            constant_friction_factor=constant_friction_factor,
        )

        # Pi-model specific: compute area and tilt angle
        self.A = math.pi * (diameter / 2) ** 2
        
        inlet_alt = getattr(inlet, 'altitude', 0.0) or 0.0
        outlet_alt = getattr(outlet, 'altitude', 0.0) or 0.0
        dz = outlet_alt - inlet_alt
        self.theta = np.arctan2(dz, length)
        self.gas_mixture_ref = self.gas_mixture # Reference to gas mixture for Pi-model calculations

    @property
    def i(self) -> int:
        """Inlet node index (Pi-model)."""
        return self.inlet_index
    
    
    @property
    def j(self) -> int:
        """Outlet node index (Pi-model)."""
        return self.outlet_index
    
    
    @property
    def L(self) -> float:
        """Length (Pi-model alias)."""
        return self.length
    
    @property
    def D(self) -> float:
        """Diameter (Pi-model alias)."""
        return self.diameter
    
    @property
    def f(self) -> float:
        """Friction factor (Pi-model)."""
        if self.constant_friction_factor is not None:
            return self.constant_friction_factor
        return self.calculate_pipe_friction_factor()
    
    @f.setter
    def f(self, value: float):
        self.constant_friction_factor = value