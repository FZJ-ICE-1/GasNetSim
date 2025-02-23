#  #!/usr/bin/env python
#  -*- coding: utf-8 -*-
#  ******************************************************************************
#    Copyright (c) 2022.
#    Developed by Yifei Lu
#    Last change on 1/17/22, 11:21 AM
#    Last change by yifei
#   *****************************************************************************
from .node import *

class ReverseFlowError(Exception):
    """Exception raised when reverse flow is detected in the compressor."""
    pass


class Compressor:
    """
    Class to formulate compressor stations.
    """

    def __init__(
        self,
        inlet: Node,
        outlet: Node,
        drive="electric",
        compression_ratio=1.1,
        thermodynamic_process="isentropic",
    ):
        self.inlet = inlet
        self.outlet = outlet
        self.drive = drive
        self.cp = inlet.gas_mixture.cp
        self.cv = inlet.gas_mixture.cv
        self.T1 = inlet.temperature
        self.r_comp = compression_ratio
        if thermodynamic_process == "isentropic":
            self.n = self.cp / self.cv
        elif thermodynamic_process == "isothermal":
            self.n = 1
        else:
            raise ValueError(
                "Only isentropic or isothermal process is currently supported."
            )

        self.mass_flow_rate = None

    def update_flow_rate(self, total_mass_flow_in, total_mass_flow_out, inlet_node_demand, outlet_node_demand):
        self.mass_flow_rate = (total_mass_flow_in - inlet_node_demand + total_mass_flow_out + outlet_node_demand) / 2

        # Check for reverse flow
        if self.mass_flow_rate < 0:
            raise ReverseFlowError(f"Reverse flow detected in compressor: {self.flow_rate} m³/s")

    def power_consumption(self):
        """
        This method calculates the power consumption of the compressor based on its thermodynamic process and compression ratio.
        Returns
        -------

        """
        return (
            self.cp * self.T1 * (self.r_comp ** ((self.n - 1) / self.n) - 1) * self.mass_flow_rate
        )

    def calculate_incoming_flows_and_derivatives(self, pipelines):
        total_mass_flow_in = 0.0
        total_derivative_in = 0.0
        total_mass_flow_out = 0.0
        total_derivative_out = 0.0

        for pipeline in pipelines:
            if pipeline.outlet == self.inlet:
                flow = pipeline.calc_flow_rate()
                derivative = pipeline.flow_rate_first_order_derivative(is_inlet=False)
                total_mass_flow_in += flow
                total_derivative_in += derivative
            elif pipeline.inlet == self.outlet:
                flow = pipeline.calc_flow_rate()
                derivative = pipeline.flow_rate_first_order_derivative(is_inlet=True)
                total_mass_flow_out += flow
                total_derivative_out += derivative

        return total_mass_flow_in, total_derivative_in, total_mass_flow_out, total_derivative_out
