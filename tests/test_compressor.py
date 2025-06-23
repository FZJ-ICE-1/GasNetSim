#!/usr/bin/env python
# -*- coding: utf-8 -*-
# ******************************************************************************
#  Copyright (c) 2025.
#  Developed by Yifei Lu
#  Compressor class unit tests
# *****************************************************************************

import unittest
from pathlib import Path
import numpy as np
from scipy.constants import bar

from GasNetSim.components.compressor import Compressor, ReverseFlowError
from GasNetSim.components.node import Node
from GasNetSim.components.pipeline import Pipeline
from GasNetSim.components.gas_mixture.typical_mixture_composition import NATURAL_GAS_gri30
import GasNetSim as gns


class TestCompressor(unittest.TestCase):
    """Unit tests for the Compressor class."""

    def setUp(self):
        """Set up test nodes and pipelines for compressor testing."""
        # Create test nodes with natural gas composition
        self.inlet_node = Node(
            node_index=1,
            pressure_pa=40 * bar,
            altitude=0,
            temperature=288.15,
            node_type="junction",
            gas_composition=NATURAL_GAS_gri30
        )

        self.outlet_node = Node(
            node_index=2,
            pressure_pa=44 * bar,  # Will be overridden by compressor
            altitude=0,
            temperature=288.15,
            node_type="junction",
            gas_composition=NATURAL_GAS_gri30
        )

        self.supply_node = Node(
            node_index=0,
            pressure_pa=50 * bar,
            altitude=0,
            temperature=288.15,
            node_type="reference",
            gas_composition=NATURAL_GAS_gri30
        )

        self.demand_node = Node(
            node_index=3,
            volumetric_flow=20.0,
            flow_type="volumetric",
            altitude=0,
            temperature=288.15,
            node_type="demand",
            gas_composition=NATURAL_GAS_gri30
        )

    def test_compressor_initialization(self):
        """Test compressor initialization with various parameters."""
        # Test default initialization
        compressor = Compressor(
            compressor_index=1,
            inlet=self.inlet_node,
            outlet=self.outlet_node
        )

        self.assertEqual(compressor.compressor_index, 1)
        self.assertEqual(compressor.inlet_index, 1)
        self.assertEqual(compressor.outlet_index, 2)
        self.assertEqual(compressor.compression_ratio, 1.1)  # Default
        self.assertEqual(compressor.efficiency, 0.85)  # Default
        self.assertEqual(compressor.drive, "electric")  # Default
        self.assertIsNone(compressor.flow_rate)
        self.assertEqual(compressor.power_consumption_value, 0.0)

    def test_compressor_custom_parameters(self):
        """Test compressor initialization with custom parameters."""
        compressor = Compressor(
            compressor_index=2,
            inlet=self.inlet_node,
            outlet=self.outlet_node,
            compression_ratio=1.5,
            efficiency=0.9,
            drive="gas_turbine",
            thermodynamic_process="isothermal"
        )

        self.assertEqual(compressor.compression_ratio, 1.5)
        self.assertEqual(compressor.efficiency, 0.9)
        self.assertEqual(compressor.drive, "gas_turbine")
        self.assertEqual(compressor.n, 1)  # Isothermal process

    def test_invalid_thermodynamic_process(self):
        """Test that invalid thermodynamic process raises error."""
        with self.assertRaises(ValueError):
            Compressor(
                compressor_index=1,
                inlet=self.inlet_node,
                outlet=self.outlet_node,
                thermodynamic_process="invalid_process"
            )

    def test_update_flow_rate(self):
        """Test compressor flow rate calculation."""
        compressor = Compressor(
            compressor_index=1,
            inlet=self.inlet_node,
            outlet=self.outlet_node,
            compression_ratio=1.2
        )

        # Test flow rate calculation
        total_flow_in = 50.0
        total_flow_out = 45.0
        inlet_node_demand = 10.0
        outlet_node_demand = 15.0

        compressor.update_flow_rate(
            total_flow_in, total_flow_out,
            inlet_node_demand, outlet_node_demand
        )

        expected_flow = (total_flow_in - inlet_node_demand + total_flow_out + outlet_node_demand) / 2
        expected_flow = (50.0 - 10.0 + 45.0 + 15.0) / 2  # = 50.0

        self.assertAlmostEqual(compressor.flow_rate, expected_flow, places=6)
        self.assertIsNotNone(compressor.mass_flow_rate)

    def test_power_calculation(self):
        """Test compressor power consumption calculation."""
        compressor = Compressor(
            compressor_index=1,
            inlet=self.inlet_node,
            outlet=self.outlet_node,
            compression_ratio=1.3,
            efficiency=0.85
        )

        # Set a flow rate
        compressor.flow_rate = 25.0
        compressor.mass_flow_rate = 25.0 * 0.8  # Approximate gas density

        power = compressor.calculate_power()

        # Power should be positive for positive flow
        self.assertGreater(power, 0)
        self.assertEqual(power, compressor.power_consumption_value)

        # Test zero flow case
        compressor.flow_rate = 0.0
        zero_power = compressor.calculate_power()
        self.assertEqual(zero_power, 0.0)

    def test_power_consumption_legacy_method(self):
        """Test legacy power_consumption() method."""
        compressor = Compressor(
            compressor_index=1,
            inlet=self.inlet_node,
            outlet=self.outlet_node
        )

        compressor.flow_rate = 20.0
        compressor.mass_flow_rate = 16.0

        power1 = compressor.calculate_power()
        power2 = compressor.power_consumption()

        self.assertEqual(power1, power2)

    def test_calculate_incoming_flows_and_derivatives(self):
        """Test calculation of incoming flows from connected pipelines."""
        compressor = Compressor(
            compressor_index=1,
            inlet=self.inlet_node,
            outlet=self.outlet_node
        )

        # Create test pipelines with proper pressure initialization
        # Set pressures for the test nodes first
        self.supply_node.pressure = 50 * bar
        self.inlet_node.pressure = 40 * bar
        self.outlet_node.pressure = 44 * bar
        self.demand_node.pressure = 35 * bar

        pipeline_in = Pipeline(
            pipeline_index=1,
            inlet=self.supply_node,
            outlet=self.inlet_node,
            diameter=0.5,
            length=1000
        )
        # Calculate actual flow rate instead of setting it manually
        pipeline_in.calc_flow_rate()

        pipeline_out = Pipeline(
            pipeline_index=2,
            inlet=self.outlet_node,
            outlet=self.demand_node,
            diameter=0.4,
            length=800
        )
        # Calculate actual flow rate instead of setting it manually
        pipeline_out.calc_flow_rate()

        pipelines = [pipeline_in, pipeline_out]

        total_flow_in, total_deriv_in, total_flow_out, total_deriv_out = \
            compressor.calculate_incoming_flows_and_derivatives(pipelines)

        # Should return actual calculated flows (not the hardcoded values)
        self.assertIsInstance(total_flow_in, float)
        self.assertIsInstance(total_flow_out, float)
        self.assertIsInstance(total_deriv_in, float)
        self.assertIsInstance(total_deriv_out, float)

        # Flows should be positive given our pressure setup
        self.assertGreater(total_flow_in, 0)
        self.assertGreater(total_flow_out, 0)

    def test_calc_flow_rate_methods(self):
        """Test various flow rate calculation methods."""
        compressor = Compressor(
            compressor_index=1,
            inlet=self.inlet_node,
            outlet=self.outlet_node
        )

        # Test with no flow set
        self.assertEqual(compressor.calc_flow_rate(), 0.0)

        # Test with flow rate set
        compressor.flow_rate = 15.5
        self.assertEqual(compressor.calc_flow_rate(), 15.5)
        self.assertEqual(compressor.calculate_stable_flow_rate(), 15.5)

    def test_gas_mixture_update(self):
        """Test gas mixture update based on flow direction."""
        compressor = Compressor(
            compressor_index=1,
            inlet=self.inlet_node,
            outlet=self.outlet_node
        )

        # Test forward flow
        compressor.flow_rate = 10.0
        compressor.update_gas_mixture()
        self.assertEqual(compressor.gas_mixture, self.inlet_node.gas_mixture)

        # Test reverse flow
        compressor.flow_rate = -5.0
        compressor.update_gas_mixture()
        self.assertEqual(compressor.gas_mixture, self.outlet_node.gas_mixture)

        # Test zero/None flow
        compressor.flow_rate = None
        compressor.update_gas_mixture()
        self.assertEqual(compressor.gas_mixture, self.inlet_node.gas_mixture)

    def test_mass_flow_calculation(self):
        """Test gas mass flow rate calculation."""
        compressor = Compressor(
            compressor_index=1,
            inlet=self.inlet_node,
            outlet=self.outlet_node
        )

        compressor.flow_rate = 20.0
        mass_flow = compressor.calc_gas_mass_flow()

        self.assertGreater(mass_flow, 0)
        # Should be volumetric flow * gas density
        expected_mass_flow = 20.0 * compressor.gas_mixture.standard_density
        self.assertAlmostEqual(mass_flow, expected_mass_flow, places=3)

    def test_slope_correction(self):
        """Test pipe slope correction (should be zero for compressors)."""
        compressor = Compressor(
            compressor_index=1,
            inlet=self.inlet_node,
            outlet=self.outlet_node
        )

        self.assertEqual(compressor.calc_pipe_slope_correction(), 0)

    def test_flow_velocity(self):
        """Test flow velocity calculation (should be zero for compressors)."""
        compressor = Compressor(
            compressor_index=1,
            inlet=self.inlet_node,
            outlet=self.outlet_node
        )

        self.assertEqual(compressor.calc_flow_velocity(), 0.0)


class TestCompressorNetworkIntegration(unittest.TestCase):
    """Integration tests for compressor within a network simulation."""

    def setUp(self):
        """Set up test paths."""
        self.examples_path = Path(__file__).parent.parent / "examples"
        self.minimal_compressor_path = self.examples_path / "minimal_network_with_compressor"

    def test_compressor_validation_network_creation(self):
        """Test that minimal compressor network loads correctly."""
        if not self.minimal_compressor_path.exists():
            self.skipTest("Minimal network with compressor example not found")

        network = gns.create_network_from_folder(self.minimal_compressor_path)

        # Basic structure validation
        self.assertIsInstance(network, gns.Network)
        self.assertGreater(len(network.nodes), 0)
        self.assertGreater(len(network.pipelines), 0)
        self.assertGreater(len(network.compressors), 0)

        # Verify compressor properties
        compressor = list(network.compressors.values())[0]
        self.assertIsInstance(compressor, Compressor)
        self.assertIn(compressor.inlet_index, network.nodes)
        self.assertIn(compressor.outlet_index, network.nodes)

    def test_compressor_simulation_convergence(self):
        """Test that network with compressor converges successfully."""
        if not self.minimal_compressor_path.exists():
            self.skipTest("Minimal network with compressor example not found")

        network = gns.create_network_from_folder(self.minimal_compressor_path)

        # Run simulation
        network.simulation(use_cuda=False, tol=1e-4)

        # Verify compressor operation
        compressor = list(network.compressors.values())[0]

        # Compressor should have positive flow rate
        self.assertIsNotNone(compressor.flow_rate)
        self.assertGreater(compressor.flow_rate, 0)

        # Power consumption should be positive
        power = compressor.power_consumption()
        self.assertGreater(power, 0)

        # Compression ratio should be maintained
        inlet_pressure = compressor.inlet.pressure
        outlet_pressure = compressor.outlet.pressure
        actual_ratio = outlet_pressure / inlet_pressure
        expected_ratio = compressor.compression_ratio

        self.assertAlmostEqual(actual_ratio, expected_ratio, places=2)

    def test_compressor_mass_balance(self):
        """Test mass balance in network with compressor."""
        if not self.minimal_compressor_path.exists():
            self.skipTest("Minimal network with compressor example not found")

        network = gns.create_network_from_folder(self.minimal_compressor_path)
        network.simulation(use_cuda=False, tol=1e-4)

        # Calculate total supply and demand
        total_supply = sum([
            -node.volumetric_flow for node in network.nodes.values()
            if node.node_type == "reference"
        ])

        total_demand = sum([
            node.volumetric_flow for node in network.nodes.values()
            if node.node_type == "demand" and node.volumetric_flow
        ])

        # Mass balance should be satisfied
        balance_error = abs(total_supply - total_demand)
        self.assertLess(balance_error, 1e-4)  # Reasonable tolerance for mass balance

    def test_compressor_jacobian_contribution(self):
        """Test that compressor contributes properly to Jacobian matrix."""
        if not self.minimal_compressor_path.exists():
            self.skipTest("Minimal network with compressor example not found")

        network = gns.create_network_from_folder(self.minimal_compressor_path)

        # Update network parameters
        init_f, init_p, init_t = network.newton_raphson_initialization()
        network.update_node_parameters(pressure=init_p, flow=init_f, temperature=init_t)
        network.update_pipeline_parameters()
        network.update_compressor_parameters()

        # Calculate Jacobian matrix
        j_mat, f_mat = network.jacobian_matrix(use_cuda=False, sparse_matrix=False)

        # Jacobian should be non-singular (determinant != 0)
        det = np.linalg.det(j_mat)
        self.assertNotEqual(det, 0.0)

        # Condition number should be reasonable (not too large)
        cond = np.linalg.cond(j_mat)
        self.assertLess(cond, 1e10)  # Reasonable condition number


if __name__ == "__main__":
    unittest.main()
