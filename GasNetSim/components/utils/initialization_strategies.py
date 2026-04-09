"""
Pressure Initialization Strategies for Gas Network Simulation

This module provides different initialization methods for nodal pressures,
particularly important for networks with compressors where poor initialization
can cause divergence.

Author: Created for GasNetSim
Date: 2025
"""

import numpy as np
from collections import deque, defaultdict
import logging

logger = logging.getLogger(__name__)


def random_uniform_initialization(network, min_pressure=70e5, max_pressure=84e5):
    """
    Original random initialization method.

    Simple but can cause convergence issues with compressors.

    Args:
        network: Network object
        min_pressure: Minimum pressure in Pa (default: 70 bar)
        max_pressure: Maximum pressure in Pa (default: 84 bar)

    Returns:
        List of initial pressures for all nodes
    """
    import random

    pressure_init = []
    for node in network.nodes.values():
        if node.pressure is not None:
            pressure_init.append(node.pressure)
        else:
            pressure_init.append(random.uniform(min_pressure, max_pressure))

    return pressure_init


def flat_reference_initialization(network):
    """
    Initialize all nodes to average reference pressure.

    Simple and stable baseline - good starting point.

    Args:
        network: Network object

    Returns:
        List of initial pressures for all nodes
    """
    # Get average reference pressure
    ref_pressures = [
        network.nodes[n].pressure
        for n in network.reference_nodes
        if network.nodes[n].pressure is not None
    ]

    if not ref_pressures:
        logger.warning("No reference pressures found, using default 70 bar")
        base_pressure = 70e5
    else:
        base_pressure = np.mean(ref_pressures)

    pressure_init = []
    for node in network.nodes.values():
        if node.pressure is not None:
            pressure_init.append(node.pressure)
        else:
            pressure_init.append(base_pressure)

    return pressure_init


def compressor_aware_initialization(network, num_passes=5, add_variation=True):
    """
    Initialize pressures respecting compressor compression ratios.

    Starts with flat reference pressure, then applies compressor constraints
    in multiple passes to handle series/parallel compressor configurations.

    For large networks, adds small random variation to avoid numerical issues
    with identical initial pressures.

    Args:
        network: Network object
        num_passes: Number of propagation passes (default: 5)
        add_variation: Add small random variation to avoid identical pressures (default: True)

    Returns:
        List of initial pressures for all nodes
    """
    import random

    # Start with flat initialization
    pressure_init = flat_reference_initialization(network)

    # For large networks, add small variation to avoid numerical instability
    # when all pressures are identical
    if add_variation and len(network.nodes) > 50:
        ref_pressure_values = [network.nodes[n].pressure for n in network.reference_nodes
                              if network.nodes[n].pressure is not None]
        avg_ref_pressure = np.mean(ref_pressure_values) if ref_pressure_values else 70e5

        # Add ±2% random variation to non-reference nodes
        for i, node_id in enumerate(sorted(network.nodes.keys())):
            if node_id not in network.reference_nodes:
                variation = random.uniform(0.98, 1.02)
                pressure_init[i] = avg_ref_pressure * variation

    if network.compressors is None:
        return pressure_init

    # Apply compressor constraints in multiple passes
    for pass_num in range(num_passes):
        pressure_changed = False

        for compressor in network.compressors.values():
            inlet_sim_idx = network.node_id_to_simulation_node_index(compressor.inlet_index)
            outlet_sim_idx = network.node_id_to_simulation_node_index(compressor.outlet_index)

            # Don't override reference node pressures
            if compressor.outlet_index in network.reference_nodes:
                continue

            inlet_pressure = pressure_init[inlet_sim_idx]
            expected_outlet = inlet_pressure * compressor.compression_ratio
            old_outlet = pressure_init[outlet_sim_idx]

            # Use damping to handle cycles and conflicts
            damping = 0.7  # Weight for new value
            pressure_init[outlet_sim_idx] = damping * expected_outlet + (1 - damping) * old_outlet

            if abs(expected_outlet - old_outlet) > 1000:  # Changed by more than 0.01 bar
                pressure_changed = True

        if not pressure_changed:
            logger.debug(f"Compressor initialization converged after {pass_num + 1} passes")
            break

    return pressure_init


def graph_based_initialization(network, drop_per_hop=0.02):
    """
    Initialize pressures based on topological distance from reference nodes.

    Key insight: The algorithm detects whether reference nodes are sources or sinks
    by checking if they have supply (negative flow) or demand (positive flow).

    For sink-type references (demand nodes):
    - Pressures INCREASE as we move away from reference nodes
    - Compressors boost pressure when traversing backward (toward sources)

    For source-type references (supply nodes):
    - Pressures DECREASE as we move away from reference nodes
    - Compressors boost pressure when traversing forward (toward sinks)

    Args:
        network: Network object
        drop_per_hop: Pressure drop per hop for pipelines/resistances (default: 0.02 = 2%)

    Returns:
        List of initial pressures for all nodes
    """
    from ..compressor import Compressor

    # Step 1: Build bidirectional adjacency graph
    adjacency = defaultdict(list)  # node_id -> [(neighbor_id, connection_type, connection_obj)]

    if network.pipelines:
        for pipeline in network.pipelines.values():
            adjacency[pipeline.inlet_index].append((pipeline.outlet_index, 'pipeline', pipeline))
            adjacency[pipeline.outlet_index].append((pipeline.inlet_index, 'pipeline', pipeline))

    if network.compressors:
        for compressor in network.compressors.values():
            # Compressor is directional: only forward propagation increases pressure
            adjacency[compressor.inlet_index].append((compressor.outlet_index, 'compressor', compressor))
            # Can traverse backward but pressure decreases
            adjacency[compressor.outlet_index].append((compressor.inlet_index, 'compressor_reverse', compressor))

    if network.resistances:
        for resistance in network.resistances.values():
            adjacency[resistance.inlet_index].append((resistance.outlet_index, 'resistance', resistance))
            adjacency[resistance.outlet_index].append((resistance.inlet_index, 'resistance', resistance))

    if hasattr(network, 'linear_resistances') and network.linear_resistances:
        for resistance in network.linear_resistances.values():
            adjacency[resistance.inlet_index].append((resistance.outlet_index, 'resistance', resistance))
            adjacency[resistance.outlet_index].append((resistance.inlet_index, 'resistance', resistance))

    if hasattr(network, 'shortpipes') and network.shortpipes:
        for shortpipe in network.shortpipes.values():
            adjacency[shortpipe.inlet_index].append((shortpipe.outlet_index, 'shortpipe', shortpipe))
            adjacency[shortpipe.outlet_index].append((shortpipe.inlet_index, 'shortpipe', shortpipe))

    # Step 2: Initialize from reference nodes and detect if they are sources or sinks
    ref_pressure_values = [network.nodes[n].pressure for n in network.reference_nodes if network.nodes[n].pressure is not None]
    if not ref_pressure_values:
        raise ValueError("Network must have at least one reference node with defined pressure")

    avg_ref_pressure = np.mean(ref_pressure_values)

    # Detect if reference nodes are sources (negative flow) or sinks (positive flow / no flow)
    # For gas networks: negative flow = supply, positive flow = demand
    ref_flows = [network.nodes[n].volumetric_flow for n in network.reference_nodes
                 if network.nodes[n].volumetric_flow is not None]

    # If reference nodes have no flow specified, they're typically sinks (demand points)
    # If most flows are positive or zero, they're sinks
    is_sink_reference = len(ref_flows) == 0 or np.mean([f for f in ref_flows]) >= 0

    if is_sink_reference:
        logger.info("Reference nodes detected as SINKS (demand points) - pressures will increase away from reference nodes")
    else:
        logger.info("Reference nodes detected as SOURCES (supply points) - pressures will decrease away from reference nodes")

    # Store all pressure estimates for each node from different paths
    node_pressure_estimates = defaultdict(list)  # node_id -> [(pressure, hops)]

    # Step 3: BFS from each reference node
    for ref_node_id in network.reference_nodes:
        ref_pressure = network.nodes[ref_node_id].pressure
        if ref_pressure is None:
            continue

        # BFS queue: (current_node, current_pressure, hops_from_ref)
        queue = deque([(ref_node_id, ref_pressure, 0)])
        visited = {ref_node_id}

        while queue:
            current_node, current_pressure, hops = queue.popleft()

            # Record this estimate
            node_pressure_estimates[current_node].append((current_pressure, hops))

            # Explore neighbors
            for neighbor_id, conn_type, conn_obj in adjacency[current_node]:
                if neighbor_id in visited:
                    continue

                # Calculate neighbor pressure based on connection type and reference type
                if is_sink_reference:
                    # Reference nodes are SINKS - when traversing AWAY from sinks,
                    # we're moving toward supply sources which need HIGHER pressure
                    if conn_type == 'compressor':
                        # Traversing forward through compressor (inlet->outlet, toward sink)
                        # This means we came FROM a higher pressure source
                        # So reverse-calculate: if outlet is current_pressure, inlet should be lower
                        neighbor_pressure = current_pressure / conn_obj.compression_ratio
                    elif conn_type == 'compressor_reverse':
                        # Traversing backward through compressor (outlet->inlet, toward source)
                        # We need higher pressure at the source/inlet
                        neighbor_pressure = current_pressure * conn_obj.compression_ratio
                    elif conn_type == 'shortpipe':
                        # Shortpipe: small pressure increase when moving toward sources
                        neighbor_pressure = current_pressure * (1 + drop_per_hop * 0.5)
                    else:
                        # Pipeline: when moving away from sink toward source, pressure should increase
                        neighbor_pressure = current_pressure * (1 + drop_per_hop * 0.5)
                else:
                    # Reference nodes are SOURCES - pressure DECREASES away from them
                    # (standard gas network with sources at high pressure)
                    if conn_type == 'compressor':
                        # Forward through compressor: pressure increases
                        neighbor_pressure = current_pressure * conn_obj.compression_ratio
                    elif conn_type == 'compressor_reverse':
                        # Backward through compressor: pressure decreases
                        neighbor_pressure = current_pressure / conn_obj.compression_ratio
                    elif conn_type == 'shortpipe':
                        # Shortpipe: no pressure change
                        neighbor_pressure = current_pressure
                    else:
                        # Pipeline or resistance: pressure drop per hop
                        neighbor_pressure = current_pressure * (1 - drop_per_hop)

                queue.append((neighbor_id, neighbor_pressure, hops + 1))
                visited.add(neighbor_id)

    # Step 4: Compute final pressure for each node by averaging estimates
    # Prefer estimates with fewer hops (weight by 1/hops)
    node_pressures = {}
    for node_id in network.nodes.keys():
        if node_id in node_pressure_estimates:
            estimates = node_pressure_estimates[node_id]
            # Weight by inverse of hops (closer paths weighted more)
            weighted_sum = sum(p / max(h, 1) for p, h in estimates)
            weight_total = sum(1 / max(h, 1) for p, h in estimates)
            node_pressures[node_id] = weighted_sum / weight_total if weight_total > 0 else avg_ref_pressure
        else:
            # Unreachable node
            logger.warning(f"Node {node_id} not reachable from any reference node, using average pressure {avg_ref_pressure/1e5:.1f} bar")
            node_pressures[node_id] = avg_ref_pressure

    # Step 5: Add small random variation for large networks to avoid numerical issues
    if len(network.nodes) > 50:
        import random
        for node_id in network.nodes.keys():
            if node_id not in network.reference_nodes:
                variation = random.uniform(0.98, 1.02)
                node_pressures[node_id] *= variation

    # Step 6: Convert to list in correct order
    pressure_init = []
    for node_id in sorted(network.nodes.keys()):
        p = node_pressures.get(node_id, avg_ref_pressure)
        if p is None or np.isnan(p):
            p = avg_ref_pressure
        pressure_init.append(p)

    return pressure_init


def hybrid_initialization(network, strategy='auto'):
    """
    Automatically select best initialization strategy based on network characteristics.

    Args:
        network: Network object
        strategy: 'auto', 'hybrid', 'random', 'flat', 'compressor_aware', or 'graph_based'

    Returns:
        List of initial pressures for all nodes
    """
    if strategy == 'auto' or strategy == 'hybrid':
        # Auto-select based on network structure
        has_compressors = network.compressors is not None and len(network.compressors) > 0
        num_nodes = len(network.nodes)

        if not has_compressors:
            # No compressors: simple flat initialization is fine
            logger.info("Using flat reference initialization (no compressors)")
            return flat_reference_initialization(network)
        elif len(network.compressors) <= 3 and num_nodes < 50:
            # Few compressors and small network: compressor-aware is sufficient
            logger.info("Using compressor-aware initialization")
            return compressor_aware_initialization(network)
        else:
            # Complex network: use graph-based approach
            logger.info("Using graph-based initialization for complex network")
            return graph_based_initialization(network)
    elif strategy == 'random':
        return random_uniform_initialization(network)
    elif strategy == 'flat':
        return flat_reference_initialization(network)
    elif strategy == 'compressor_aware':
        return compressor_aware_initialization(network)
    elif strategy == 'graph_based':
        return graph_based_initialization(network)
    else:
        raise ValueError(f"Unknown initialization strategy: {strategy}")


def resistance_weighted_initialization(network):
    """
    Initialize pressures by propagating from known-pressure nodes using a
    resistance-weighted pressure drop heuristic.

    For each connection the pressure of an unknown node is estimated as:

        p_outlet = p_inlet * (1 - 0.5 * (res / max_res))

    Shortpipes (zero resistance) get no pressure drop. Iteration continues
    until no pressure value changes (converged assignment).

    Args:
        network: Network object

    Returns:
        List of initial pressures for all nodes (simulation index order)
    """
    import copy

    resistance = []
    if network.pipelines is not None:
        resistance += [
            [x.inlet_index, x.outlet_index, x.resistance]
            for x in network.pipelines.values()
        ]
    if network.resistances is not None:
        resistance += [
            [x.inlet_index, x.outlet_index, x.resistance]
            for x in network.resistances.values()
        ]
    if network.linear_resistances is not None:
        resistance += [
            [x.inlet_index, x.outlet_index, x.resistance]
            for x in network.linear_resistances.values()
        ]
    if network.shortpipes is not None:
        resistance += [
            [x.inlet_index, x.outlet_index, 0]
            for x in network.shortpipes.values()
        ]

    max_resistance = max(r[2] for r in resistance)

    pressure_init = [node.pressure for node in network.nodes.values()]
    pressure_init_old = []

    while pressure_init != pressure_init_old:
        pressure_init_old = copy.deepcopy(pressure_init)
        for inlet_node_id, outlet_node_id, res in resistance:
            i = network.node_id_to_simulation_node_index(inlet_node_id)
            j = network.node_id_to_simulation_node_index(outlet_node_id)
            drop = 0.5 * (res / max_resistance) if max_resistance > 0 else 0
            if pressure_init[i] is None and pressure_init[j] is None:
                pass
            elif pressure_init[j] is None or pressure_init[i] == pressure_init[j]:
                pressure_init[j] = pressure_init[i] * (1 - drop)
            elif pressure_init[i] is None and pressure_init[j] is not None:
                pressure_init[i] = pressure_init[j] / (1 - drop) if drop < 1 else pressure_init[j]

    return pressure_init


# Dictionary of available strategies for easy access
INITIALIZATION_STRATEGIES = {
    'resistance_weighted': resistance_weighted_initialization,
    'random': random_uniform_initialization,
    'flat': flat_reference_initialization,
    'compressor_aware': compressor_aware_initialization,
    'graph_based': graph_based_initialization,
    'hybrid': hybrid_initialization,
}


def get_initialization_strategy(strategy_name='hybrid'):
    """
    Get initialization function by name.

    Args:
        strategy_name: Name of strategy ('random', 'flat', 'compressor_aware',
                      'graph_based', or 'hybrid')

    Returns:
        Initialization function
    """
    if strategy_name not in INITIALIZATION_STRATEGIES:
        raise ValueError(
            f"Unknown initialization strategy: {strategy_name}. "
            f"Available: {list(INITIALIZATION_STRATEGIES.keys())}"
        )
    return INITIALIZATION_STRATEGIES[strategy_name]
