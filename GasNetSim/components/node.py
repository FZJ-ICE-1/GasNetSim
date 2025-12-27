#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
#   ******************************************************************************
#     Copyright (c) 2024.
#     Developed by Yifei Lu
#     Last change on 10/17/24, 4:42 PM
#     Last change by yifei
#    *****************************************************************************
from scipy.constants import bar

from .gas_mixture.typical_mixture_composition import NATURAL_GAS_gri30
from .gas_mixture.gas_mixture import GasMixture
from ..utils.exception import InitializationError


class Node:
    """
    Class to formulate gas transmission network nodes.
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
    ):
        """
        Initial method
        :param node_index: Node index
        :param flow: Gas volumetric flow [sm3/s] or energy flow [MJ/s]
        :param pressure_pa: Gas nodal pressure [Pa]
        :param temperature: Gas nodal temperature [K]
        :param altitude: Elevation of the network node [m]
        :param gas_composition: Gas nodal composition
        :param node_type: Type of node
        :param altitude: Altitude of the network node [m]
        :param longitude: Longitude of node [degrees]
        :param latitude: Latitude of node [degrees]
        """
        self.index = node_index
        if gas_composition is not None:
            self.gas_composition = gas_composition
        else:
            self.gas_composition = NATURAL_GAS_gri30
        self.pressure = pressure_pa
        if pressure_pa is not None:
            self.pressure_bar = pressure_pa / bar
        if temperature is not None:
            self.temperature = temperature
        else:
            self.temperature = 288.15
        if altitude is not None:
            self.altitude = altitude
        else:
            self.altitude = 0
        if node_type is not None:
            self.node_type = node_type
        else:
            self.node_type = "demand"
        self.longitude = longitude
        self.latitude = latitude
        # flow type
        if flow_type is not None:
            self.flow_type = flow_type
        else:
            self.flow_type = "volumetric"

        try:
            self.gas_mixture = GasMixture(
                composition=self.gas_composition,
                temperature=self.temperature,
                pressure=self.pressure,
            )
        except (TypeError, AttributeError):
            # If pressure or temperature is missing for some nodes
            self.gas_mixture = GasMixture(
                composition=self.gas_composition, temperature=288.15, pressure=50 * bar
            )

        self.volumetric_flow = volumetric_flow
        self.energy_flow = energy_flow

        if self.flow_type == "volumetric" and volumetric_flow is not None:
            try:
                self.convert_volumetric_to_energy_flow()
            except TypeError:
                self.energy_flow = None
        elif self.flow_type == "energy" and energy_flow is not None:
            try:
                self.convert_energy_to_volumetric_flow()
            except TypeError:
                self.volumetric_flow = None
        else:
            if pressure_pa is None:
                raise InitializationError(f"Either pressure or flow should be known: node {self.index}.")

    def update_gas_mixture(self):
        try:
            self.gas_mixture = GasMixture(composition=self.get_mole_fraction(),
                                          temperature=self.temperature,
                                          pressure=self.pressure)
        except (TypeError, AttributeError):
            self.gas_mixture = GasMixture(composition=NATURAL_GAS_gri30,
                                          temperature=288.15,
                                          pressure=50 * bar)

    def get_mole_fraction(self):
        """
        Get mole fraction of the gas composition at node
        :return: Gas mole fraction
        """
        # mole_fraction = dict()
        # thermo_mixture = Mixture(zs=self.gas_mixture.composition,
        #                          T=self.gas_mixture.temperature,
        #                          P=self.gas_mixture.pressure)
        # for i in range(len(thermo_mixture.zs)):
        #     gas = thermo_mixture.IDs[i]
        #     mole_fraction[gas] = thermo_mixture.zs[i]
        # return mole_fraction
        return self.gas_mixture.composition

    def convert_energy_to_volumetric_flow(self):
        """
        Convert energy flow rate (MW) into volumetric flow rate (sm^3/s)
        :return:
        """
        # HHV = calc_heating_value(self.gas_mixture)
        # HHV = self.gas_mixture.heating_value(hhv=True, parameter="mass")
        self.volumetric_flow = (
            self.energy_flow / self.gas_mixture.HHV_J_per_sm3 * 1e6
        )  # sm3/s

    def convert_volumetric_to_energy_flow(self):
        """
        Convert volumetric flow rate (sm^3/s) into energy flow rate (MW)
        :return:
        """
        # HHV = self.gas_mixture.heating_value(hhv=True, parameter="mass")
        self.energy_flow = (
            self.volumetric_flow * self.gas_mixture.HHV_J_per_sm3 / 1e6
        )  # MJ/s


if __name__ == "__main__":
    from GasNetSim.components.gas_mixture import *
    from GasNetSim.components.gas_mixture import *
    from GasNetSim.utils.exception import InitializationError

    Node(node_index=1, volumetric_flow=10, pressure_pa=None, temperature=300)
