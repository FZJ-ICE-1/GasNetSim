#!/usr/bin/env python
# coding: utf-8

# **********************************************************************************************************************
# This test script serves to confirm that the Network class maintains its functionality post-modifications.
# The test script contains 3 test functions:
#   1. test_network_volume_flow_rate_balance() - to ensure the volume flow rate balance within a network.
#   2. test_network_energy_flow_balance() - to ensure the energy flow balance within a network.
#   3. test_network_composition_balance() - to ensure the accuracy of gas composition within a network simulation.
# **********************************************************************************************************************

import logging
import os
from pathlib import Path

from numpy.testing import assert_almost_equal, assert_allclose

import GasNetSim as gns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def find_git_root(path):
    """
    Find the root path of the Git repository starting from the given path.

    Args:
    - path: The starting directory path to search from.

    Returns:
    - The root path of the Git repository or None if not found.
    """
    # Traverse up the directory tree until finding the .git folder
    while path != "/":
        if os.path.isdir(os.path.join(path, ".git")):
            return path
        path = os.path.dirname(path)
    return None


def test_network_volume_flow_rate_balance():
    """
    Test to ensure the volume flow rate balance within a network.
    Calculates the total inflow and outflow across all nodes in the network to verify conservation
    of volume flow rates.
    """

    # Find the current absolute path
    test_directory_path = os.path.abspath(os.getcwd())
    # Find the root path of the Git repository
    root_path = find_git_root(test_directory_path)
    new_path = os.path.join(root_path, "examples", "Irish13")

    # Create a network instance with Irish13
    # Initialize the network with nodes and connections from a CSV file in the current directory
    # network = gns.create_network_from_csv(Path('../examples/Irish13/.'))
    network = gns.create_network_from_folder(Path(new_path))

    # Simulate the network to compute the pressures and flows
    network.simulation(tol=0.0000001)

    # Define the expected final pressure values for comparison
    # expected_final_pressure = [7000000., 7000000., 7000000., 6933669.72289829,
    #                            6835569.60774993, 6623690.36661754, 6605743.43571843,
    #                            6602395.62670504, 6600915.11259321, 6592750.20799817,
    #                            6795230.64383087, 6791480.37767578, 6753737.3583671]

    # Retrieve the final pressure values from the simulated network nodes
    # final_pressure = [node.pressure for node in network.nodes.values()]

    # Calculate total inflow and outflow over the entire Network
    # Initialize lists to store node information
    node_indices = []
    inlet_flows = []
    outlet_flows = []

    # Calculate initial flow rates from the nodes
    initial_flows = {
        node_index: node.volumetric_flow if node.volumetric_flow is not None else 0
        for node_index, node in network.nodes.items()
    }

    # Iterate through nodes in the network to gather inlet and outlet flow information
    for node_index, node in network.nodes.items():
        inlet_flow = 0
        outlet_flow = 0

        # Iterate through connections to find flows related to the current node
        for connection in network.connections.values():
            if connection.outlet_index == node_index:
                inlet_flow += connection.flow_rate
            elif connection.inlet_index == node_index:
                outlet_flow += connection.flow_rate

        # Add initial flow to the outlet flow for each node
        initial_flow = initial_flows.get(node_index, 0)
        if initial_flow <= 0:
            inlet_flow -= initial_flow
        else:
            outlet_flow += initial_flow

        # Append node information to the lists
        node_indices.append(node_index)
        inlet_flows.append(inlet_flow)
        outlet_flows.append(outlet_flow)

    # Calculate total inflow and outflow for the entire network
    total_inflow = sum(inlet_flows)
    total_outflow = sum(outlet_flows)

    # Check if the final pressure values from the simulation match the expected values
    # assert_almost_equal(final_pressure, expected_final_pressure)
    assert_almost_equal(inlet_flows, outlet_flows)
    assert_almost_equal(total_inflow, total_outflow)

    # If the assertion passes, print a message indicating that the test passed
    logger.info(
        f"Test passed: Results match the expected values for volume_flow_rate_balance."
    )


def test_network_energy_flow_balance():
    """
    Test to ensure the energy flow balance within a network.
    Calculates the energy flowing in and out of each node in the network to verify energy conservation.
    """
    # Find the current absolute path
    test_directory_path = os.path.abspath(os.getcwd())
    # Find the root path of the Git repository
    root_path = find_git_root(test_directory_path)
    new_path = os.path.join(root_path, "examples", "Irish13")

    # Create a network instance with Irish13
    # Initialize the network with nodes and connections from a CSV file in the current directory
    # network = gns.create_network_from_csv(Path('../examples/Irish13/.'))
    network = gns.create_network_from_folder(Path(new_path))

    # Simulate the network to compute the pressures and flows
    network.simulation(tol=0.0000001)

    # Calculate energy flow at inlet and outlet for each node over the entire Network
    # Calculate initial flow rates from the nodes
    initial_volume_flows = {
        node_index: node.volumetric_flow if node.volumetric_flow is not None else 0
        for node_index, node in network.nodes.items()
    }

    # Iterate through nodes in the network to gather inlet and outlet flow information
    for node_index, node in network.nodes.items():
        inlet_flow = 0
        outlet_flow = 0
        inlet_energy = 0
        outlet_energy = 0

        # Iterate through connections to find flows related to the current node
        for connection in network.connections.values():
            if connection.outlet_index == node_index:
                inlet_flow += connection.flow_rate
            elif connection.inlet_index == node_index:
                outlet_flow += connection.flow_rate

        # Add initial flow based on sign to the inlet or outlet flow for each node
        initial_flow = initial_volume_flows.get(node_index, 0)
        if initial_flow <= 0:
            inlet_flow -= initial_flow
        else:
            outlet_flow += initial_flow

        # Retrieve pressure values from the simulated network nodes
        pressure = node.pressure

        # Calculate inlet and outlet energy based on pressure and flows
        inlet_energy += pressure * inlet_flow
        outlet_energy += pressure * outlet_flow
        # print(f"inlet energy = {inlet_energy}, outlet energy = {outlet_energy}")
        assert_allclose(inlet_energy, outlet_energy)
        # print(f"inlet = {inlet_flow}, outlet = {outlet_flow}")

    # If the assertion passes, print a message indicating that the test passed
    logger.info(
        f"Test passed: Results match the expected values for energy_flow_rate_balance."
    )


def test_network_composition_balance():
    """
    Test to ensure the accuracy of gas composition within a network simulation.
    Calculates the total volumetric flow rate and component wise flow rate to verify the composition balance.
    """
    # Find the current absolute path
    test_directory_path = os.path.abspath(os.getcwd())
    # Find the root path of the Git repository
    root_path = find_git_root(test_directory_path)
    new_path = os.path.join(root_path, "examples", "Irish13")

    # Create a network instance with Irish13
    # Initialize the network with nodes and connections from a CSV file in the current directory
    # network = gns.create_network_from_csv(Path('../examples/Irish13/.'))
    network = gns.create_network_from_folder(Path(new_path))

    # Simulate the network to compute the pressures and flows
    network.simulation(tol=0.0000001)

    # Calculate the composition balance for each pipeline using volumetric flow rates
    for i, pipeline in network.pipelines.items():
        # Calculate the total volumetric flow rate
        volumetric_flow_rate = (
            pipeline.flow_rate
        )  # Assuming flow rate is already volumetric

        # Calculate the volumetric flow rate of each component (assuming mole_fraction is equivalent to volume fraction)
        component_flow_rates = {
            component: mole_fraction * volumetric_flow_rate
            for component, mole_fraction in pipeline.gas_mixture.composition.items()
        }
        # print(component_flow_rates)

        # Calculate the total component volumetric flow rate
        total_component_volumetric_flow_rate = sum(component_flow_rates.values())

        # Check if the total component volumetric flow rate equals the total volumetric flow rate within a tolerance
        assert_almost_equal(total_component_volumetric_flow_rate, volumetric_flow_rate)
        # print([total_component_volumetric_flow_rate, volumetric_flow_rate])

    # If the assertion passes, print a message indicating that the test passed
    logger.info(
        f"Test passed: Results match the expected values for composition_balance."
    )
