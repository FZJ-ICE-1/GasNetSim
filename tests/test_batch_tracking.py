#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
#   ******************************************************************************
#     Copyright (c) 2025.
#     Developed by Yifei Lu
#     Last change on 1/12/25, 2:38 PM
#     Last change by yifei
#    *****************************************************************************
import unittest
import numpy as np
from scipy.constants import bar

from GasNetSim.components import Node, Pipeline
from GasNetSim.components.utils.utils import batch_tracking, clean_boundary_batches


class MockPipeline:
    def __init__(self, length):
        """
        Initializes a mock pipeline for batch tracking simulations.

        :param length: Length of the pipeline [m].
        """
        self.length = length  # Pipeline length
        self.batch_location_history = np.array([])  # Locations of batches
        self.composition_history = np.array([[]])  # Compositions of batches
        self.inlet_composition = None  # Current inlet composition
        self.outlet_composition = None  # Current outlet composition
        self.inflow_composition = None  # Composition of the gas inflow
        self.outflow_composition = None  # Composition of the gas outflow


class TestBatchTracking(unittest.TestCase):
    def setUp(self):
        self.length = 4.5  # Pipeline length in meters
        self.inlet_composition = np.array([0] * 21)
        self.outlet_composition = np.array([1] * 21)
        self.connection = MockPipeline(self.length)

    def test_zero_velocity(self):
        self.connection.batch_location_history = [1, 2]
        self.connection.composition_history = [np.array([0] * 21), np.array([1] * 21)]
        velocity = 0
        inflow_composition = self.inlet_composition if velocity >= 0 else self.outlet_composition
        outflow_composition = self.outlet_composition if velocity >= 0 else self.inlet_composition

        # Call batch_tracking with zero velocity
        batch_location_history, composition_history, outflow_composition = (
            batch_tracking(
                1,
                velocity,
                self.length,
                inflow_composition,
                outflow_composition,
                self.connection.batch_location_history,
                self.connection.composition_history,
            )
        )

        self.assertEqual([1, 2], batch_location_history)
        np.testing.assert_array_equal([0] * 21, composition_history[0])
        np.testing.assert_array_equal([1] * 21, composition_history[1])
        # self.assertIsNone(inlet_comp)
        # self.assertIsNone(outlet_comp)

    def test_no_batches(self):
        velocity = 1.0
        inflow_composition = self.inlet_composition if velocity >= 0 else self.outlet_composition
        outflow_composition = self.outlet_composition if velocity >= 0 else self.inlet_composition

        batch_location_history, composition_history, outflow_composition = (
            batch_tracking(
                1,
                velocity,
                self.length,
                inflow_composition,
                outflow_composition,
                [],
                [],
            )
        )

        self.assertEqual([0], batch_location_history)
        np.testing.assert_array_equal(composition_history[0], self.inlet_composition)
        # self.assertIsNone(inlet_comp)
        # self.assertIsNone(outlet_comp)

    def test_forward_flow_boundary(self):
        self.connection.batch_location_history = [5]
        self.connection.composition_history = [np.array([0] * 21)]

        batch_location_history, composition_history, outflow_composition = (
            clean_boundary_batches(
                self.connection.outflow_composition,
                self.connection.batch_location_history,
                self.connection.composition_history,
                self.length,
                1,
            )
        )

        self.assertEqual(batch_location_history, [])
        self.assertEqual(len(composition_history), 0)
        np.testing.assert_array_equal(outflow_composition, np.array([0] * 21))
        # self.assertIsNone(inlet_comp)

    def test_reverse_flow_boundary(self):
        self.connection.batch_location_history = [-1]
        self.connection.composition_history = [np.array([0] * 21)]

        batch_location_history, composition_history, outflow_composition = (
            clean_boundary_batches(
                self.connection.outflow_composition,
                self.connection.batch_location_history,
                self.connection.composition_history,
                self.length,
                -1,
            )
        )

        self.assertEqual(batch_location_history, [])
        self.assertEqual(len(composition_history), 0)
        np.testing.assert_array_equal(outflow_composition, np.array([0] * 21))
        # self.assertIsNone(outlet_comp)

    def test_multiple_batches_boundary(self):
        self.connection.batch_location_history = [5, 6, -1]
        self.connection.composition_history = [
            np.array([0] * 21),
            np.array([1] * 21),
            np.array([2] * 21),
        ]

        batch_location_history, composition_history, outflow_composition = (
            clean_boundary_batches(
                self.connection.outflow_composition,
                self.connection.batch_location_history,
                self.connection.composition_history,
                self.length,
                1,
            )
        )

        self.assertEqual([-1], batch_location_history)
        self.assertEqual(1, len(composition_history))
        np.testing.assert_array_equal(np.array([1] * 21), outflow_composition)
        np.testing.assert_array_equal(np.array([2] * 21), composition_history[0])

    def test_alternating_flow(self):
        self.connection.batch_location_history = [1, 0]
        self.connection.composition_history = [np.array([5] * 21), np.array([6] * 21)]
        velocity = 1.0
        inflow_composition = self.inlet_composition if velocity >= 0 else self.outlet_composition
        outflow_composition = self.outlet_composition if velocity >= 0 else self.inlet_composition

        # Forward flow
        batch_location_history, composition_history, outflow_composition = (
            batch_tracking(
                1,
                velocity,
                self.length,
                inflow_composition,
                outflow_composition,
                self.connection.batch_location_history,
                self.connection.composition_history,
            )
        )
        self.assertEqual(batch_location_history, [2, 1, 0])

        velocity = -1.0
        inflow_composition = self.inlet_composition if velocity >= 0 else self.outlet_composition
        outflow_composition = self.outlet_composition if velocity >= 0 else self.inlet_composition
        # Reverse flow
        batch_location_history, composition_history, outflow_composition = (
            batch_tracking(
                1,
                velocity,
                self.length,
                inflow_composition,
                outflow_composition,
                batch_location_history,
                composition_history,
            )
        )

        self.assertEqual([4.5, 1], batch_location_history)
        np.testing.assert_array_equal(outflow_composition, np.array([6] * 21))

    def test_batch_exactly_at_boundary(self):
        self.connection.batch_location_history = [4.5]
        self.connection.composition_history = [np.array([0] * 21)]

        batch_location_history, composition_history, outflow_composition = (
            clean_boundary_batches(
                self.connection.outflow_composition,
                self.connection.batch_location_history,
                self.connection.composition_history,
                self.length,
                1,
            )
        )

        self.assertEqual([], batch_location_history)
        np.testing.assert_array_equal(np.array([0] * 21), outflow_composition)
        # np.testing.assert_array_equal(np.array([0] * 21), outflow_composition)

    def test_rapid_alternating_flow(self):
        self.connection.batch_location_history = [1, 0]
        self.connection.composition_history = [np.array([0] * 21), np.array([1] * 21)]
        velocity = 1.0
        inflow_composition = self.inlet_composition if velocity >= 0 else self.outlet_composition
        outflow_composition = self.outlet_composition if velocity >= 0 else self.inlet_composition

        # Forward flow
        batch_location_history, composition_history, _ = batch_tracking(
            1,
            velocity,
            self.length,
            inflow_composition,
            outflow_composition,
            self.connection.batch_location_history,
            self.connection.composition_history,
        )

        velocity = -1.0
        inflow_composition = self.inlet_composition if velocity >= 0 else self.outlet_composition
        outflow_composition = self.outlet_composition if velocity >= 0 else self.inlet_composition
        # Reverse flow immediately after forward flow
        batch_location_history, composition_history, _ = batch_tracking(
            1,
            -1.0,
            self.length,
            inflow_composition,
            outflow_composition,
            batch_location_history,
            composition_history,
        )

        self.assertEqual(
            [4.5, 1],
            batch_location_history,
        )
        self.assertEqual(2, len(composition_history))

    def test_large_number_of_batches(self):
        self.connection.batch_location_history = list(range(50))
        self.connection.composition_history = [np.array([i] * 21) for i in range(50)]
        velocity = 1.0
        inflow_composition = self.inlet_composition if velocity >= 0 else self.outlet_composition
        outflow_composition = self.outlet_composition if velocity >= 0 else self.inlet_composition

        batch_location_history, composition_history, _ = batch_tracking(
            1,
            velocity,
            100,
            inflow_composition,
            outflow_composition,
            self.connection.batch_location_history,
            self.connection.composition_history,
        )

        self.assertEqual(51, len(batch_location_history), 51)
        self.assertEqual(51, len(composition_history), 51)

    def test_empty_compositions(self):
        self.connection.batch_location_history = [1, 2]
        self.connection.composition_history = []
        velocity = 1.0
        inflow_composition = self.inlet_composition if velocity >= 0 else self.outlet_composition
        outflow_composition = self.outlet_composition if velocity >= 0 else self.inlet_composition

        with self.assertRaises(ValueError):
            batch_tracking(
                1,
                1.0,
                self.length,
                inflow_composition,
                outflow_composition,
                self.connection.batch_location_history,
                self.connection.composition_history,
            )
