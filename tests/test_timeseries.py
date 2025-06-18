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
from GasNetSim.simulation.timeseries import read_profiles, check_profiles
from pathlib import Path


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
            pipeline_index=1,
            inlet=self.node1,
            outlet=self.node2,
            diameter=0.5,
            length=1000,
            efficiency=0.85,
        )
        self.pipe2 = Pipeline(
            pipeline_index=2,
            inlet=self.node2,
            outlet=self.node3,
            diameter=0.5,
            length=1500,
            efficiency=0.85,
        )
        self.pipe3 = Pipeline(
            pipeline_index=3,
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


class TestProfileReading(unittest.TestCase):
    """Test cases for profile reading and validation functions."""
    
    def setUp(self):
        """Set up test paths to profile test data."""
        self.test_data_path = Path(__file__).parent / "data" / "profiles"
        
    def test_read_basic_profiles(self):
        """Test reading basic profiles without time column."""
        profiles = read_profiles(self.test_data_path / "basic_profiles.csv")
        
        # Check structure
        self.assertIsInstance(profiles, pd.DataFrame)
        self.assertEqual(list(profiles.columns), [2, 3])
        self.assertEqual(len(profiles), 5)
        
        # Check values
        self.assertEqual(profiles.iloc[0, 0], 20)  # First row, first column
        self.assertEqual(profiles.iloc[0, 1], 30)  # First row, second column
        self.assertEqual(profiles.iloc[-1, 0], 40)  # Last row, first column
        self.assertEqual(profiles.iloc[-1, 1], 50)  # Last row, second column
        
    def test_read_profiles_with_time_integer(self):
        """Test reading profiles with integer time column."""
        profiles = read_profiles(self.test_data_path / "profiles_with_time.csv")
        
        # Check structure
        self.assertIsInstance(profiles, pd.DataFrame)
        self.assertIn("time", profiles.columns)
        self.assertEqual(set(profiles.columns), {"time", 2, 3})
        
        # Check time values
        self.assertEqual(profiles["time"].iloc[0], 0)
        self.assertEqual(profiles["time"].iloc[1], 3600)
        
    def test_read_profiles_with_datetime(self):
        """Test reading profiles with datetime time column."""
        profiles = read_profiles(self.test_data_path / "profiles_with_datetime.csv")
        
        # Check structure  
        self.assertIsInstance(profiles, pd.DataFrame)
        self.assertIn("time", profiles.columns)
        
        # Check that time column is parsed correctly
        # Note: the exact parsing depends on pandas version, but it should be time-like
        self.assertTrue(len(profiles["time"]) > 0)
        
    def test_read_profiles_with_unnamed_index(self):
        """Test reading profiles with 'Unnamed: 0' index column."""
        profiles = read_profiles(self.test_data_path / "profiles_with_unnamed_index.csv")
        
        # Check that Unnamed: 0 column is handled
        self.assertNotIn("Unnamed: 0", profiles.columns)
        self.assertEqual(list(profiles.columns), [2, 3])
        self.assertEqual(len(profiles), 5)
        
    def test_read_empty_profiles_file(self):
        """Test reading empty profiles file raises appropriate error."""
        with self.assertRaises(ValueError) as cm:
            read_profiles(self.test_data_path / "empty_profiles.csv")
        self.assertIn("contains no data", str(cm.exception))
            
    def test_read_nonexistent_file(self):
        """Test reading nonexistent file raises FileNotFoundError."""
        with self.assertRaises(FileNotFoundError):
            read_profiles(self.test_data_path / "nonexistent_file.csv")
            
    def test_check_profiles_without_time(self):
        """Test check_profiles with DataFrame without time column."""
        df = pd.DataFrame({2: [20, 25, 30], 3: [30, 35, 40]})
        result = check_profiles(df)
        
        # Should return DataFrame unchanged
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result.columns), [2, 3])
        
    def test_check_profiles_with_integer_time(self):
        """Test check_profiles with integer time column."""
        df = pd.DataFrame({
            "time": [0, 3600, 7200],
            2: [20, 25, 30],
            3: [30, 35, 40]
        })
        result = check_profiles(df)
        
        # Should convert time to datetime and set as index
        self.assertIsInstance(result, pd.DataFrame)
        self.assertNotIn("time", result.columns)  # Should be moved to index
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result.index))
        self.assertEqual(list(result.columns), [2, 3])
        
    def test_check_profiles_with_datetime_time(self):
        """Test check_profiles with datetime time column."""
        df = pd.DataFrame({
            "time": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 01:00:00"]),
            2: [20, 25],
            3: [30, 35]
        })
        result = check_profiles(df)
        
        # Should set time as index
        self.assertIsInstance(result, pd.DataFrame)
        self.assertNotIn("time", result.columns)  # Should be moved to index
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result.index))
        
    def test_check_profiles_with_invalid_time(self):
        """Test check_profiles with invalid time column raises error."""
        df = pd.DataFrame({
            "time": ["invalid", "time", "data"],
            2: [20, 25, 30],
            3: [30, 35, 40]
        })
        with self.assertRaises(ValueError) as cm:
            check_profiles(df)
        self.assertIn("time", str(cm.exception).lower())
        
    def test_integrated_read_and_check_profiles(self):
        """Test integrated workflow of reading and checking profiles."""
        # Read profiles with time
        profiles = read_profiles(self.test_data_path / "profiles_with_time.csv")
        
        # Check/validate profiles  
        checked_profiles = check_profiles(profiles)
        
        # Verify integrated result
        self.assertIsInstance(checked_profiles, pd.DataFrame)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(checked_profiles.index))
        self.assertEqual(list(checked_profiles.columns), [2, 3])
        self.assertEqual(len(checked_profiles), 5)


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
                profile_type="volumetric",  # Fix: specify that profiles contain volumetric flows
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
