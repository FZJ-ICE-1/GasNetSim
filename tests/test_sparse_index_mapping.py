#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Test sparse index mapping functionality to ensure networks with non-consecutive 
node indices work correctly.
"""

import numpy as np
import pandas as pd
from unittest import TestCase
import unittest

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from GasNetSim.components.node import Node
from GasNetSim.components.pipeline import Pipeline
from GasNetSim.components.network import Network
from GasNetSim.components.gas_mixture.gas_mixture import GasMixture
from GasNetSim.components.gas_mixture.typical_mixture_composition import NATURAL_GAS_gri30


class TestSparseIndexMapping(TestCase):
    """Test cases for sparse node index mapping functionality."""
    
    def setUp(self):
        """Set up test fixtures with sparse node indices."""
        self.base_composition = NATURAL_GAS_gri30
        
    def _create_test_nodes_sparse(self):
        """Create test nodes with sparse indices [1, 5, 10, 15]."""
        nodes = {}
        sparse_indices = [1, 5, 10, 15]
        
        for i, node_idx in enumerate(sparse_indices):
            if i == 0:  # First node is reference with fixed pressure
                nodes[node_idx] = Node(
                    node_index=node_idx,
                    pressure_pa=5e5,  # 5 bar
                    temperature=288.15,
                    gas_composition=self.base_composition,
                    node_type="reference"
                )
            else:  # Other nodes are demand nodes
                nodes[node_idx] = Node(
                    node_index=node_idx,
                    volumetric_flow=-10.0,  # Demand
                    temperature=288.15,
                    gas_composition=self.base_composition,
                    node_type="junction"
                )
        return nodes
        
    def _create_test_pipelines_sparse(self, nodes):
        """Create test pipelines connecting sparse nodes."""
        pipelines = {}
        
        # Pipeline 1: connects nodes 1 -> 5
        pipelines[1] = Pipeline(
            pipeline_index=1,
            inlet=nodes[1],
            outlet=nodes[5],
            diameter=0.5,  # 0.5 m
            length=10000,  # 10 km
            efficiency=0.95,
            roughness=0.0001
        )
        
        # Pipeline 2: connects nodes 5 -> 10
        pipelines[2] = Pipeline(
            pipeline_index=2,
            inlet=nodes[5],
            outlet=nodes[10],
            diameter=0.4,  # 0.4 m
            length=15000,  # 15 km
            efficiency=0.95,
            roughness=0.0001
        )
        
        # Pipeline 3: connects nodes 10 -> 15
        pipelines[3] = Pipeline(
            pipeline_index=3,
            inlet=nodes[10],
            outlet=nodes[15],
            diameter=0.3,  # 0.3 m
            length=8000,  # 8 km
            efficiency=0.95,
            roughness=0.0001
        )
        
        return pipelines
        
    def test_index_mapping_creation(self):
        """Test that index mapping is created correctly for sparse indices."""
        nodes = self._create_test_nodes_sparse()
        network = Network(nodes=nodes)
        
        # Check mapping dictionaries
        expected_node_id_to_sim = {1: 0, 5: 1, 10: 2, 15: 3}
        expected_sim_to_node_id = {0: 1, 1: 5, 2: 10, 3: 15}
        
        self.assertEqual(network._node_id_to_simulation_node_index, expected_node_id_to_sim)
        self.assertEqual(network._simulation_node_index_to_node_id, expected_sim_to_node_id)
        self.assertEqual(network._max_node_id, 15)
        
    def test_node_id_to_simulation_conversion(self):
        """Test node ID to simulation index conversion."""
        nodes = self._create_test_nodes_sparse()
        network = Network(nodes=nodes)
        
        # Test valid conversions
        self.assertEqual(network.node_id_to_simulation_node_index(1), 0)
        self.assertEqual(network.node_id_to_simulation_node_index(5), 1)
        self.assertEqual(network.node_id_to_simulation_node_index(10), 2)
        self.assertEqual(network.node_id_to_simulation_node_index(15), 3)
        
        # Test invalid index
        with self.assertRaises(ValueError):
            network.node_id_to_simulation_node_index(99)
            
    def test_simulation_to_node_id_conversion(self):
        """Test simulation index to node ID conversion."""
        nodes = self._create_test_nodes_sparse()
        network = Network(nodes=nodes)
        
        # Test valid conversions
        self.assertEqual(network.simulation_node_index_to_node_id(0), 1)
        self.assertEqual(network.simulation_node_index_to_node_id(1), 5)
        self.assertEqual(network.simulation_node_index_to_node_id(2), 10)
        self.assertEqual(network.simulation_node_index_to_node_id(3), 15)
        
        # Test invalid index
        with self.assertRaises(ValueError):
            network.simulation_node_index_to_node_id(99)
            
    def test_matrix_dimensions(self):
        """Test that matrices have correct dimensions with sparse indices."""
        nodes = self._create_test_nodes_sparse()
        pipelines = self._create_test_pipelines_sparse(nodes)
        network = Network(nodes=nodes, pipelines=pipelines)
        
        # Matrix should be 4x4 (number of nodes), not 15x15 (max index)
        connection_matrix = network.create_connection_matrix()
        self.assertEqual(connection_matrix.shape, (4, 4))
        
        # Check mapping matrix
        mapping_matrix = network.mapping_of_connections()
        self.assertEqual(mapping_matrix.shape, (4, 4))
        
        # Check incidence matrix
        incidence_matrix = network.create_incidence_matrix()
        self.assertEqual(incidence_matrix.shape, (4, 3))  # 4 nodes, 3 pipelines
        
    def test_connection_matrix_correctness(self):
        """Test that connection matrix correctly maps sparse indices to dense."""
        nodes = self._create_test_nodes_sparse()
        pipelines = self._create_test_pipelines_sparse(nodes)
        network = Network(nodes=nodes, pipelines=pipelines)
        
        connection_matrix = network.create_connection_matrix()
        
        # Check connections exist in dense matrix coordinates
        # Pipeline 1: nodes 1->5 maps to dense 0->1
        self.assertEqual(connection_matrix[0, 1], 1)
        self.assertEqual(connection_matrix[1, 0], 1)
        
        # Pipeline 2: nodes 5->10 maps to dense 1->2
        self.assertEqual(connection_matrix[1, 2], 1)
        self.assertEqual(connection_matrix[2, 1], 1)
        
        # Pipeline 3: nodes 10->15 maps to dense 2->3
        self.assertEqual(connection_matrix[2, 3], 1)
        self.assertEqual(connection_matrix[3, 2], 1)
        
        # Check no spurious connections
        self.assertEqual(connection_matrix[0, 2], 0)  # No direct 1->10 connection
        self.assertEqual(connection_matrix[0, 3], 0)  # No direct 1->15 connection
        
    def test_network_simulation_with_sparse_indices(self):
        """Test that network simulation works with sparse node indices."""
        nodes = self._create_test_nodes_sparse()
        pipelines = self._create_test_pipelines_sparse(nodes)
        network = Network(nodes=nodes, pipelines=pipelines)
        
        # This should not raise any IndexError
        try:
            network.simulation(max_iter=10, tol=0.1)
            simulation_success = True
        except (IndexError, KeyError) as e:
            simulation_success = False
            self.fail(f"Simulation failed with sparse indices: {e}")
            
        self.assertTrue(simulation_success)
        
        # Check that all nodes have updated pressures
        for node_idx in [1, 5, 10, 15]:
            self.assertIsNotNone(network.nodes[node_idx].pressure)
            self.assertGreater(network.nodes[node_idx].pressure, 0)
            
    def test_jacobian_matrix_with_sparse_indices(self):
        """Test jacobian matrix calculation with sparse indices."""
        nodes = self._create_test_nodes_sparse()
        pipelines = self._create_test_pipelines_sparse(nodes)
        network = Network(nodes=nodes, pipelines=pipelines)
        
        # Initialize network for jacobian calculation
        network.newton_raphson_initialization()
        
        # This should not raise IndexError
        try:
            jacobian_mat, flow_mat = network.jacobian_matrix()
            jacobian_success = True
        except (IndexError, KeyError) as e:
            jacobian_success = False
            self.fail(f"Jacobian calculation failed with sparse indices: {e}")
            
        self.assertTrue(jacobian_success)
        
        # Check matrix dimensions (should account for reference nodes)
        n_junction_nodes = len(network.junction_nodes)
        self.assertEqual(jacobian_mat.shape[0], n_junction_nodes)
        self.assertEqual(jacobian_mat.shape[1], n_junction_nodes)
        
    def test_consecutive_vs_sparse_indices(self):
        """Test that sparse and consecutive indexing give same results."""
        
        # Create network with consecutive indices [1, 2, 3, 4]
        consecutive_nodes = {}
        for i in range(1, 5):
            if i == 1:
                consecutive_nodes[i] = Node(
                    node_index=i,
                    pressure_pa=5e5,
                    temperature=288.15,
                    gas_composition=self.base_composition,
                    node_type="reference"
                )
            else:
                consecutive_nodes[i] = Node(
                    node_index=i,
                    volumetric_flow=-10.0,
                    temperature=288.15,
                    gas_composition=self.base_composition,
                    node_type="junction"
                )
                
        consecutive_pipelines = {
            1: Pipeline(pipeline_index=1, inlet=consecutive_nodes[1], outlet=consecutive_nodes[2], diameter=0.5, length=10000, efficiency=0.95, roughness=0.0001),
            2: Pipeline(pipeline_index=2, inlet=consecutive_nodes[2], outlet=consecutive_nodes[3], diameter=0.4, length=15000, efficiency=0.95, roughness=0.0001),
            3: Pipeline(pipeline_index=3, inlet=consecutive_nodes[3], outlet=consecutive_nodes[4], diameter=0.3, length=8000, efficiency=0.95, roughness=0.0001)
        }
        
        # Create networks
        consecutive_network = Network(nodes=consecutive_nodes, pipelines=consecutive_pipelines)
        sparse_nodes = self._create_test_nodes_sparse()
        sparse_pipelines = self._create_test_pipelines_sparse(sparse_nodes)
        sparse_network = Network(nodes=sparse_nodes, pipelines=sparse_pipelines)
        
        # Run simulations
        consecutive_network.simulation(max_iter=50, tol=0.001)
        sparse_network.simulation(max_iter=50, tol=0.001)
        
        # Extract results (skip reference node)
        consecutive_pressures = [consecutive_network.nodes[i].pressure for i in [2, 3, 4]]
        sparse_pressures = [sparse_network.nodes[i].pressure for i in [5, 10, 15]]
        
        # Results should be very similar (within 1% tolerance)
        for cp, sp in zip(consecutive_pressures, sparse_pressures):
            self.assertAlmostEqual(cp, sp, delta=abs(cp * 0.01))
            
    def test_edge_cases(self):
        """Test edge cases for sparse index mapping."""
        
        # Test single node
        single_node = {42: Node(42, pressure_pa=5e5, temperature=288.15,
                               gas_composition=self.base_composition,
                               node_type="reference")}
        single_network = Network(nodes=single_node)
        self.assertEqual(single_network.node_id_to_simulation_node_index(42), 0)
        self.assertEqual(single_network.simulation_node_index_to_node_id(0), 42)
        
        # Test very large sparse indices
        large_indices = {1000: Node(1000, pressure_pa=5e5, temperature=288.15,
                                  gas_composition=self.base_composition,
                                  node_type="reference"),
                        2000: Node(2000, volumetric_flow=-10.0, temperature=288.15,
                                 gas_composition=self.base_composition,
                                 node_type="junction")}
        large_pipeline = {1: Pipeline(pipeline_index=1, inlet=large_indices[1000], outlet=large_indices[2000], diameter=0.5, length=10000, efficiency=0.95, roughness=0.0001)}
        large_network = Network(nodes=large_indices, pipelines=large_pipeline)
        
        # Should create 2x2 matrix, not 2000x2000
        connection_matrix = large_network.create_connection_matrix()
        self.assertEqual(connection_matrix.shape, (2, 2))


if __name__ == '__main__':
    unittest.main()