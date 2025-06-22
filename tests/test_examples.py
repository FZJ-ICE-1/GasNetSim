#!/usr/bin/env python
# -*- coding: utf-8 -*-
#******************************************************************************
#  Copyright (c) 2025.
#  Developed by Yifei Lu
#  Test script to validate all example scripts
#*****************************************************************************

import unittest
import subprocess
import sys
from pathlib import Path
import importlib.util


class TestExamples(unittest.TestCase):
    """Test that all example scripts run successfully."""

    def setUp(self):
        """Set up test paths."""
        self.examples_path = Path(__file__).parent.parent / "examples"

    def test_irish13_example(self):
        """Test that Irish13 example runs without error."""
        irish13_path = self.examples_path / "Irish13"
        if not irish13_path.exists():
            self.skipTest("Irish13 example not found")
        
        script_path = irish13_path / "Irish13.py"
        if not script_path.exists():
            self.skipTest("Irish13.py script not found")
        
        # Test by importing and running (safer than subprocess for tests)
        original_cwd = Path.cwd()
        try:
            # Change to example directory
            import os
            os.chdir(irish13_path)
            
            # Import the module
            spec = importlib.util.spec_from_file_location("irish13", script_path)
            irish13_module = importlib.util.module_from_spec(spec)
            
            # This should not raise any exceptions
            spec.loader.exec_module(irish13_module)
            
        except Exception as e:
            self.fail(f"Irish13 example failed to run: {e}")
        finally:
            os.chdir(original_cwd)

    def test_minimal_compressor_example(self):
        """Test that minimal compressor network example runs without error."""
        compressor_path = self.examples_path / "minimal_network_with_compressor"
        if not compressor_path.exists():
            self.skipTest("Minimal compressor example not found")
        
        script_path = compressor_path / "minimal_network_with_compressor.py"
        if not script_path.exists():
            self.skipTest("minimal_network_with_compressor.py script not found")
        
        # Test by importing and running
        original_cwd = Path.cwd()
        try:
            import os
            os.chdir(compressor_path)
            
            # Import the module
            spec = importlib.util.spec_from_file_location("compressor_example", script_path)
            compressor_module = importlib.util.module_from_spec(spec)
            
            # This should not raise any exceptions
            spec.loader.exec_module(compressor_module)
            
        except Exception as e:
            self.fail(f"Minimal compressor example failed to run: {e}")
        finally:
            os.chdir(original_cwd)

    def test_examples_have_required_files(self):
        """Test that example directories have the required CSV files."""
        
        # Test Irish13
        irish13_path = self.examples_path / "Irish13"
        if irish13_path.exists():
            self.assertTrue((irish13_path / "Irish13_nodes.csv").exists())
            self.assertTrue((irish13_path / "Irish13_pipelines.csv").exists())
        
        # Test minimal compressor network
        compressor_path = self.examples_path / "minimal_network_with_compressor"
        if compressor_path.exists():
            # Check for CSV files (either with prefix or standard names)
            has_nodes = (compressor_path / "nodes.csv").exists() or \
                       (compressor_path / "minimal_network_with_compressor_nodes.csv").exists()
            has_pipelines = (compressor_path / "pipelines.csv").exists() or \
                           (compressor_path / "minimal_network_with_compressor_pipelines.csv").exists()
            has_compressors = (compressor_path / "compressors.csv").exists() or \
                             (compressor_path / "minimal_network_with_compressor_compressors.csv").exists()
            
            self.assertTrue(has_nodes, "Compressor example missing nodes CSV")
            self.assertTrue(has_pipelines, "Compressor example missing pipelines CSV")
            self.assertTrue(has_compressors, "Compressor example missing compressors CSV")

    def test_h2_injection_example_structure(self):
        """Test that H2 injection example has proper structure (if present)."""
        h2_path = self.examples_path / "Irish13_h2_injections"
        if not h2_path.exists():
            self.skipTest("Irish13 H2 injection example not found")
        
        # Should have nodes, pipelines, and shortpipes
        self.assertTrue((h2_path / "Irish13_h2_injection_nodes.csv").exists())
        self.assertTrue((h2_path / "Irish13_h2_injection_pipelines.csv").exists())
        self.assertTrue((h2_path / "Irish13_h2_injection_shortpipes.csv").exists())

    def test_resistance_example_structure(self):
        """Test that resistance example has proper structure (if present)."""
        resistance_path = self.examples_path / "Irish13_resistance"
        if not resistance_path.exists():
            self.skipTest("Irish13 resistance example not found")
        
        # Should have nodes and resistance files
        self.assertTrue((resistance_path / "Irish13_nodes.csv").exists())
        self.assertTrue((resistance_path / "Irish13_resistance.csv").exists())


if __name__ == "__main__":
    unittest.main()