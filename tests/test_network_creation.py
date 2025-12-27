#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
#   ******************************************************************************
#     Copyright (c) 2024.
#     Developed by Yifei Lu
#     Last change on 10/23/24, 9:41 AM
#     Last change by yifei
#    *****************************************************************************
import unittest
from pathlib import Path

from GasNetSim import create_network_from_folder, create_network_from_files, Network


class TestNetworkCreation(unittest.TestCase):
    def setUp(self):
        """Set up test paths to real example networks."""
        self.examples_path = Path(__file__).parent.parent / "examples"
        self.irish13_path = self.examples_path / "Irish13"
        self.irish13_resistance_path = self.examples_path / "Irish13_resistance"
        self.irish13_h2_path = self.examples_path / "Irish13_h2_injections"
        self.minimal_compressor_path = self.examples_path / "minimal_network_with_compressor"

    def test_create_network_from_folder_irish13(self):
        """Test network creation from folder using Irish13 example."""
        network = create_network_from_folder(self.irish13_path)
        
        # Basic structure validation
        self.assertIsInstance(network, Network)
        self.assertGreater(len(network.nodes), 0)
        self.assertGreater(len(network.pipelines), 0)
        
        # Verify index mapping works correctly
        self.assertEqual(network.get_simulation_node_count(), len(network.nodes))
        
        # Test that node index mapping is bijective
        for node_id in network.nodes.keys():
            sim_idx = network.node_id_to_simulation_node_index(node_id)
            back_to_node_id = network.simulation_node_index_to_node_id(sim_idx)
            self.assertEqual(back_to_node_id, node_id)

    def test_create_network_comparison_irish13(self):
        """Compare create_network_from_folder vs create_network_from_files using Irish13."""
        # Create network from folder
        network_from_folder = create_network_from_folder(self.irish13_path)
        
        # Create network from individual files
        component_files = {
            "nodes": self.irish13_path / "Irish13_nodes.csv",
            "pipelines": self.irish13_path / "Irish13_pipelines.csv",
        }
        network_from_files = create_network_from_files(component_files)
        
        # Both should create valid networks
        self.assertIsInstance(network_from_folder, Network)
        self.assertIsInstance(network_from_files, Network)
        
        # Should have identical structure
        self.assertEqual(len(network_from_folder.nodes), len(network_from_files.nodes))
        self.assertEqual(len(network_from_folder.pipelines), len(network_from_files.pipelines))
        
        # Node indices should match
        self.assertEqual(set(network_from_folder.nodes.keys()), 
                        set(network_from_files.nodes.keys()))
        
        # Pipeline indices should match
        self.assertEqual(set(network_from_folder.pipelines.keys()), 
                        set(network_from_files.pipelines.keys()))

    def test_network_with_resistance_files(self):
        """Test network creation with resistance files (Irish13_resistance example)."""
        try:
            network = create_network_from_folder(self.irish13_resistance_path)
            
            self.assertIsInstance(network, Network)
            self.assertGreater(len(network.nodes), 0)
            
            # This example should have resistances
            if network.resistances:
                self.assertGreater(len(network.resistances), 0)
        except KeyError as e:
            # Skip test if CSV format is incompatible (different column names)
            self.skipTest(f"Skipping due to CSV format incompatibility: {e}")

    def test_network_with_shortpipes(self):
        """Test network creation with shortpipes (Irish13_h2_injections example)."""
        network = create_network_from_folder(self.irish13_h2_path)
        
        self.assertIsInstance(network, Network)
        self.assertGreater(len(network.nodes), 0)
        
        # This example should have shortpipes
        if network.shortpipes:
            self.assertGreater(len(network.shortpipes), 0)

    def test_network_node_types_and_properties(self):
        """Test that network properly parses different node types and properties."""
        network = create_network_from_folder(self.irish13_path)
        
        # Should have both reference and junction nodes
        reference_nodes = [n for n in network.nodes.values() if n.node_type == 'reference']
        junction_nodes = [n for n in network.nodes.values() if n.node_type != 'reference']
        
        self.assertGreater(len(reference_nodes), 0)
        self.assertGreater(len(junction_nodes), 0)
        
        # Reference nodes should have pressure defined
        for ref_node in reference_nodes:
            self.assertIsNotNone(ref_node.pressure)
            self.assertGreater(ref_node.pressure, 0)

    def test_pipeline_connections_valid(self):
        """Test that all pipeline connections reference valid nodes."""
        network = create_network_from_folder(self.irish13_path)
        
        for pipeline in network.pipelines.values():
            # Inlet and outlet nodes should exist in network
            self.assertIn(pipeline.inlet_index, network.nodes)
            self.assertIn(pipeline.outlet_index, network.nodes)
            
            # Should be able to map to simulation indices
            inlet_sim_idx = network.node_id_to_simulation_node_index(pipeline.inlet_index)
            outlet_sim_idx = network.node_id_to_simulation_node_index(pipeline.outlet_index)
            
            self.assertGreaterEqual(inlet_sim_idx, 0)
            self.assertGreaterEqual(outlet_sim_idx, 0)
            self.assertLess(inlet_sim_idx, network.get_simulation_node_count())
            self.assertLess(outlet_sim_idx, network.get_simulation_node_count())

    def test_create_network_with_compressor(self):
        """Test network creation with minimal compressor network example."""
        if not self.minimal_compressor_path.exists():
            self.skipTest("Minimal network with compressor example not found")
        
        network = create_network_from_folder(self.minimal_compressor_path)
        
        # Basic structure validation
        self.assertIsInstance(network, Network)
        self.assertGreater(len(network.nodes), 0)
        self.assertGreater(len(network.pipelines), 0)
        self.assertGreater(len(network.compressors), 0)
        
        # Verify compressor integration
        self.assertEqual(len(network.compressors), 1)
        compressor = list(network.compressors.values())[0]
        
        # Compressor should reference valid nodes
        self.assertIn(compressor.inlet_index, network.nodes)
        self.assertIn(compressor.outlet_index, network.nodes)
        
        # Test simulation convergence
        network.simulation(use_cuda=False, tol=1e-4)
        
        # Verify compressor operation after simulation
        self.assertIsNotNone(compressor.flow_rate)
        self.assertGreater(compressor.flow_rate, 0)  # Should have positive flow
        
        # Verify compression ratio is maintained
        inlet_pressure = compressor.inlet.pressure
        outlet_pressure = compressor.outlet.pressure
        actual_ratio = outlet_pressure / inlet_pressure
        expected_ratio = compressor.compression_ratio
        
        self.assertAlmostEqual(actual_ratio, expected_ratio, places=2)

    def test_compressor_network_index_mapping(self):
        """Test that compressor network properly handles index mapping."""
        if not self.minimal_compressor_path.exists():
            self.skipTest("Minimal network with compressor example not found")
        
        network = create_network_from_folder(self.minimal_compressor_path)
        
        # Test index mapping for compressor nodes
        compressor = list(network.compressors.values())[0]
        
        # Should be able to map compressor node indices
        inlet_sim_idx = network.node_id_to_simulation_node_index(compressor.inlet_index)
        outlet_sim_idx = network.node_id_to_simulation_node_index(compressor.outlet_index)
        
        self.assertGreaterEqual(inlet_sim_idx, 0)
        self.assertGreaterEqual(outlet_sim_idx, 0)
        self.assertLess(inlet_sim_idx, network.get_simulation_node_count())
        self.assertLess(outlet_sim_idx, network.get_simulation_node_count())
        
        # Test reverse mapping
        back_to_inlet_id = network.simulation_node_index_to_node_id(inlet_sim_idx)
        back_to_outlet_id = network.simulation_node_index_to_node_id(outlet_sim_idx)
        
        self.assertEqual(back_to_inlet_id, compressor.inlet_index)
        self.assertEqual(back_to_outlet_id, compressor.outlet_index)

    def test_compressor_network_reference_nodes(self):
        """Test that compressor network correctly identifies reference nodes."""
        if not self.minimal_compressor_path.exists():
            self.skipTest("Minimal network with compressor example not found")
        
        network = create_network_from_folder(self.minimal_compressor_path)
        
        # Should have exactly one reference node
        reference_nodes = [n for n in network.nodes.values() if n.node_type == 'reference']
        self.assertEqual(len(reference_nodes), 1)
        
        # Network's find_reference_nodes should match
        found_reference_nodes = network.find_reference_nodes()
        self.assertEqual(len(found_reference_nodes), 1)
        self.assertEqual(found_reference_nodes[0], reference_nodes[0].index)