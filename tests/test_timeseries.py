#!/usr/bin/env python
# -*- coding: utf-8 -*-
#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
#   ******************************************************************************
#     Copyright (c) 2025.
#     Developed by Yifei Lu
#     Last change on 1/5/25, 11:00 PM
#     Last change by yifei
#    *****************************************************************************

import unittest
from scipy.constants import bar
from collections import OrderedDict
import tempfile
import os
import pandas as pd
from parameterized import parameterized

from GasNetSim import (
    Node,
    Pipeline,
    Network,
    GasMixture,
    validate_results_to_save,
    run_time_series,
    save_time_series_results,
)


class BaseTestNetwork(unittest.TestCase):
    """Base class for setting up a 3-node network used in multiple test cases."""

    def setUp(self):
        # Define gas mixture
        gas_mixture = GasMixture(
            composition=OrderedDict({"methane": 0.9, "hydrogen": 0.1}),
            temperature=300,
            pressure=50 * bar,
        )

        # Create nodes
        self.node1 = Node(
            node_index=1,
            pressure_pa=50 * bar,
            gas_composition=gas_mixture.composition,
            temperature=300,
            altitude=100,
            node_type="reference",
        )
        self.node2 = Node(
            node_index=2,
            pressure_pa=None,
            volumetric_flow=20,
            gas_composition=gas_mixture.composition,
            temperature=288.15,
            altitude=40,
        )
        self.node3 = Node(
            node_index=3,
            pressure_pa=None,
            volumetric_flow=30,
            gas_composition=gas_mixture.composition,
            temperature=288.15,
            altitude=50,
        )

        # Create pipelines
        self.pipe1 = Pipeline(
            inlet=self.node1,
            outlet=self.node2,
            diameter=0.5,
            length=1000,
            efficiency=0.85,
        )
        self.pipe2 = Pipeline(
            inlet=self.node2,
            outlet=self.node3,
            diameter=0.5,
            length=1500,
            efficiency=0.85,
        )
        self.pipe3 = Pipeline(
            inlet=self.node1,
            outlet=self.node3,
            diameter=0.5,
            length=1500,
            efficiency=0.85,
        )

        # Create network
        self.network = Network(
            nodes={1: self.node1, 2: self.node2, 3: self.node3},
            pipelines={1: self.pipe1, 2: self.pipe2, 3: self.pipe3},
        )
        self.network.reference_nodes = [1]  # Node 1 is the reference node


class MockProfiles:
    """Mock profiles to simulate time-series data for tests."""

    def __init__(self, values=None, nodes=None):
        self.values = values or [
            [20, 30],
            [25, 35],
            [30, 40],
            [35, 45],
            [40, 50],
        ]
        self.df = pd.DataFrame(self.values, columns=[2, 3])

    def __getitem__(self, key):
        return self.df[key]

    def iloc(self, t):
        return self.df.iloc[t]

    @property
    def index(self):
        return self.df.index


class TestValidateResultsToSave(BaseTestNetwork):
    """Test cases for validating results keys."""

    def test_valid_keys(self):
        """Ensure no error is raised for valid result keys."""
        validate_results_to_save(["nodal_pressure", "pipeline_flowrate"])

    def test_invalid_keys(self):
        """Ensure ValueError is raised for invalid result keys."""
        with self.assertRaises(ValueError) as context:
            validate_results_to_save(["invalid_key", "nodal_pressure"])
        self.assertIn("Unrecognized keys", str(context.exception))
        self.assertIn("invalid_key", str(context.exception))


class TestSaveTimeSeriesResults(BaseTestNetwork):
    """Test cases for saving time-series results."""

    def test_save_valid_results(self):
        """Test saving results with valid keys."""
        results_to_save = ["nodal_pressure", "pipeline_flowrate"]
        results = dict([(k, []) for k in results_to_save])
        updated_results = save_time_series_results(
            self.network, results, results_to_save
        )

        self.assertIn("nodal_pressure", updated_results)
        self.assertEqual(len(updated_results["nodal_pressure"]), 1)
        self.assertIn("pipeline_flowrate", updated_results)
        self.assertEqual(len(updated_results["pipeline_flowrate"]), 1)

    def test_save_with_invalid_key(self):
        """Ensure ValueError is raised when saving with invalid keys."""
        results_to_save = ["invalid_key"]
        results = dict([(k, []) for k in results_to_save])
        with self.assertRaises(ValueError):
            save_time_series_results(self.network, results, results_to_save)

    def test_empty_profiles(self):
        """Ensure ValueError is raised for empty profiles."""
        profiles = pd.DataFrame([], columns=[2, 3])
        with self.assertRaises(ValueError):
            run_time_series(
                network=self.network,
                profiles=profiles,
                results_to_save=["nodal_pressure"],
                output_format="csv",
                output_filename="test_output",
            )

    def test_simulation_and_file_output(self):
        """Test simulation with valid profiles and verify file output."""
        results_to_save = ["nodal_pressure", "pipeline_flowrate"]
        with tempfile.TemporaryDirectory() as tmpdirname:
            results = run_time_series(
                network=self.network,
                profiles=pd.DataFrame(
                    [
                        [20, 30],
                        [25, 35],
                        [30, 40],
                        [35, 45],
                        [40, 50],
                    ],
                    columns=[2, 3],
                ),
                results_to_save=results_to_save,
                output_format="csv",
                output_filename=os.path.join(tmpdirname, "test_output"),
            )

            self.assertIn("nodal_pressure", results)
            self.assertIn("pipeline_flowrate", results)
            self.assertEqual(len(results["nodal_pressure"]), 5)
            self.assertEqual(len(results["pipeline_flowrate"]), 5)


class TestParameterizedSaveTimeSeriesResults(BaseTestNetwork):
    """Parameterized tests for saving time-series results."""

    @parameterized.expand(
        [
            (["nodal_pressure", "pipeline_flowrate"], True),
            (["invalid_key"], False),
        ]
    )
    def test_save_results(self, results_to_save, is_valid):
        """Test saving results with valid and invalid keys."""
        results = dict([(k, []) for k in results_to_save])
        if is_valid:
            updated_results = save_time_series_results(
                self.network, results, results_to_save
            )
            for key in results_to_save:
                self.assertIn(key, updated_results)
        else:
            with self.assertRaises(ValueError):
                save_time_series_results(self.network, results, results_to_save)
