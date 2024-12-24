#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
#   ******************************************************************************
#     Copyright (c) 2024.
#     Developed by Yifei Lu
#     Last change on 10/23/24, 9:41 AM
#     Last change by yifei
#    *****************************************************************************
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

from GasNetSim import create_network_from_folder, create_network_from_files, Network


class TestNetworkCreation(unittest.TestCase):
    @patch("GasNetSim.components.utils.create_network.read_nodes")
    @patch("GasNetSim.components.utils.create_network.read_pipelines")
    @patch("GasNetSim.components.utils.create_network.read_compressors")
    @patch("GasNetSim.components.utils.create_network.read_resistances")
    @patch("GasNetSim.components.utils.create_network.read_linear_resistances")
    @patch("GasNetSim.components.utils.create_network.read_shortpipes")
    @patch("pathlib.Path.glob")
    def test_create_network_comparison(
        self,
        mock_glob,
        mock_read_shortpipes,
        mock_read_linear_resistances,
        mock_read_resistances,
        mock_read_compressors,
        mock_read_pipelines,
        mock_read_nodes,
    ):
        # Setup mock behavior
        mock_read_nodes.return_value = MagicMock()
        mock_read_pipelines.return_value = MagicMock()
        mock_read_compressors.return_value = MagicMock()
        mock_read_resistances.return_value = MagicMock()
        mock_read_linear_resistances.return_value = MagicMock()
        mock_read_shortpipes.return_value = MagicMock()
        mock_glob.return_value = [
            Path("/fake/path/nodes.csv"),
            Path("/fake/path/pipelines.csv"),
        ]

        # Setup paths and files
        path = Path("/fake/path")
        component_files = {
            "nodes": Path("/fake/path/nodes.csv"),
            "pipelines": Path("/fake/path/pipelines.csv"),
        }

        # Create networks using both functions
        network_from_folder = create_network_from_folder(path)
        network_from_files = create_network_from_files(component_files)

        # Assertions
        self.assertIsInstance(network_from_folder, Network)
        self.assertIsInstance(network_from_files, Network)
        self.assertEqual(network_from_folder.nodes, network_from_files.nodes)
        self.assertEqual(network_from_folder.pipelines, network_from_files.pipelines)
        self.assertEqual(
            network_from_folder.compressors, network_from_files.compressors
        )
        self.assertEqual(
            network_from_folder.resistances, network_from_files.resistances
        )
        self.assertEqual(
            network_from_folder.linear_resistances,
            network_from_files.linear_resistances,
        )
        self.assertEqual(network_from_folder.shortpipes, network_from_files.shortpipes)
