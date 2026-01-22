"""
Dynamic Node - subclass of GasNetSim Node with Pi-model helpers.

Author: Yetkin Civan Serin
"""

import numpy as np
from typing import Optional
from GasNetSim.components.node import Node

# Fallback only (used if gas_mixture.standard_density is missing)
DEFAULT_SM3_TO_KG = 0.6816


class DynamicNode(Node):
    """
    Node subclass for dynamic Pi-model simulation.
    
    Inherits all GasNetSim Node functionality and adds:
    - global_index: index accounting for virtual nodes offset
    - virtual_index: for supply nodes, the associated virtual node index
    - demand_profile: time-varying demand array [kg/s]
    - mass_flow property: demand in kg/s (for Pi-model)
    """
    def __init__(
        self,
        node_index,
        volumetric_flow=None,
        energy_flow=None,
        pressure_pa=None,
        temperature=288.15,
        altitude=0,
        gas_composition=None,
        node_type="demand",
        flow_type=None,
        longitude=None,
        latitude=None,
        # Dynamic simulation specific
        global_index: Optional[int] = None,
        virtual_index: Optional[int] = None,
        sm3_to_kg: float = DEFAULT_SM3_TO_KG,
    ):
        # Parent constructor
        super().__init__(
            node_index=node_index,
            volumetric_flow=volumetric_flow,
            energy_flow=energy_flow,
            pressure_pa=pressure_pa,
            temperature=temperature,
            altitude=altitude,
            gas_composition=gas_composition,
            node_type=node_type,
            flow_type=flow_type,
            longitude=longitude,
            latitude=latitude,
        )

        # Dynamic simulation properties
        self.global_index = global_index if global_index is not None else node_index
        self.virtual_index = virtual_index
        self.sm3_to_kg = sm3_to_kg

        # Time-varying demand profile [kg/s]
        self._demand_profile: Optional[np.ndarray] = None


    # Pi-model properties

    @property
    def is_supply_node(self) -> bool:
        
        return str(self.node_type).lower() in {"reference", "supply", "slack"}

    @property
    def is_demand_node(self) -> bool:
        
        if self.is_supply_node:
            return False
        return self.node_type == "demand"

    @property
    def is_junction_node(self) -> bool:
        
        return (not self.is_supply_node) and (not self.is_demand_node)


    @property
    def mass_flow(self) -> float:
        """
        Base mass flow [kg/s] derived from volumetric_flow [sm3/s].

        Positive = withdrawal/demand, Negative = injection/supply.
        """
        if self.volumetric_flow is None:
            return 0.0

        # Use mixture standard density if available
        rho_std = getattr(getattr(self, "gas_mixture", None), "standard_density", None)
        if rho_std is not None:
            return self.volumetric_flow * float(rho_std)

        # Fallback constant only if mixture not available
        return self.volumetric_flow * float(self.sm3_to_kg)


    # Demand profile utilities

    @property
    def demand_profile(self) -> Optional[np.ndarray]:
        """Time-varying demand profile [kg/s]."""
        return self._demand_profile

    @demand_profile.setter
    def demand_profile(self, profile: Optional[np.ndarray]):
        """Set time-varying demand profile [kg/s]."""
        self._demand_profile = profile.copy() if profile is not None else None

    def get_demand_at_step(self, step: int) -> float:
        """
        Get demand at specific time step [kg/s].
        
        Returns profile value if set, otherwise base mass_flow.
        """
        if self._demand_profile is not None and 0 <= step < len(self._demand_profile):
            return float(self._demand_profile[step])
        return float(self.mass_flow)

    def set_demand_profile_from_volumetric(self, profile_sm3: np.ndarray):
        """
        Set demand profile from volumetric flow array [sm³/s].
        
        Converts to mass flow [kg/s] internally.
        """
        rho_std = getattr(getattr(self, "gas_mixture", None), "standard_density", None)
        factor = float(rho_std) if rho_std is not None else float(self.sm3_to_kg)
        self._demand_profile = profile_sm3 * factor

    def create_constant_profile(self, n_steps: int) -> np.ndarray:
        """
        Create constant demand profile for n_steps.
        
        Returns array of mass_flow repeated n_steps times.
        """
        self._demand_profile = np.full(n_steps, float(self.mass_flow))
        return self._demand_profile

    # keep pressure_bar synced in dynamic runs

    def set_pressure(self, pressure_pa: float):
        self.pressure = pressure_pa
        # base Node sets pressure_bar only once; keep it consistent
        try:
            from scipy.constants import bar
            self.pressure_bar = pressure_pa / bar
        except Exception:
            pass

    @property
    def supply_pressure(self) -> Optional[float]:
        return self.pressure if self.is_supply_node else None

    def __repr__(self) -> str:
        node_info = f"DynamicNode(idx={self.index}, global={self.global_index}"
        if self.is_supply_node:
            node_info += f", type=reference, P={self.pressure/1e5:.1f}bar"
            if self.virtual_index is not None:
                node_info += f", virtual={self.virtual_index}"
        elif self.is_demand_node:
            node_info += f", type=demand, m={self.mass_flow:.3f}kg/s"
        else:
            node_info += ", type=junction"
        return node_info + ")"