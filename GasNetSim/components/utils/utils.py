#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
#   ******************************************************************************
#     Copyright (c) 2025.
#     Developed by Yifei Lu
#     Last change on 1/10/25, 3:41 PM
#     Last change by yifei
#    *****************************************************************************
import math
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import seaborn as sns
from scipy import sparse

from .cuda_support import create_matrix_of_zeros
from ..pipeline import Pipeline


# try:
#     import cupy as cp
#     import cupy.sparse.linalg as cpsplinalg
# except ImportError:
#     # logging.warning(f"CuPy is not installed or not available!")
#     print(f"CuPy is not installed or not available!")


def create_connection_matrix(
        n_nodes: int,
        components: dict,
        component_type: int,
        use_cuda=False,
        sparse_matrix: bool = False,
):
    row_ind = list()
    col_ind = list()
    data = list()

    if not sparse_matrix:
        cnx = create_matrix_of_zeros(
            n_nodes, use_cuda=use_cuda, sparse_matrix=sparse_matrix
        )

    for comp in components.values():
        i = comp.inlet_index - 1
        j = comp.outlet_index - 1
        if sparse_matrix:
            row_ind.append(i)
            col_ind.append(j)
            data.append(component_type)
        else:
            cnx[i][j] = component_type
            cnx[j][i] = component_type

    if sparse_matrix:
        cnx = sparse.coo_matrix((data, (row_ind, col_ind)))
    return cnx


def levenberg_marquardt_damping_factor(m, s, b):
    return 10 ** (m * math.log10(s + b))


def delete_matrix_rows_and_columns(matrix, to_remove, use_cuda=False):
    new_matrix = matrix

    if use_cuda:
        new_matrix = cp.delete(new_matrix, to_remove, 0)  # delete rows
        new_matrix = cp.delete(new_matrix, to_remove, 1)  # delete columns
    else:
        new_matrix = np.delete(new_matrix, to_remove, 0)  # delete rows
        new_matrix = np.delete(new_matrix, to_remove, 1)  # delete columns

    return new_matrix


def jacobian_matrix_condition_number(matrix):
    print(f"The condition number of the matrix is {np.linalg.cond(matrix)}.")


def print_n_largest_absolute_values(n, values):
    sorted_values = sorted([abs(x) for x in values])
    print(sorted_values[-n::-1])
    return None


def check_batch_and_composition_lengths(batch_location_history, composition_history):
    """
    Checks that the lengths of batch_location_history and composition_history are equal.

    :param batch_location_history: List of batch locations.
    :param composition_history: List of batch compositions.
    :raises ValueError: If the lengths of the two lists are not equal.
    """
    if len(batch_location_history) != len(composition_history):
        raise ValueError(
            f"Mismatch in lengths: batch_location_history ({len(batch_location_history)}) "
            f"and composition_history ({len(composition_history)})."
        )


def batch_tracking(
        time_step,
        velocity,
        length,
        inflow_composition,
        outflow_composition,
        batch_location_history,
        composition_history,
):
    """
    Batch tracking algorithm using lists for dynamic operations and NumPy arrays for calculations.

    :param time_step: Time step for the simulation [s].
    :param velocity: Gas flow velocity [m/s], positive for forward, negative for reverse.
    :param length: Pipeline length [m].
    :param inflow_composition: 21-element numpy array for inflow gas composition.
    :param outflow_composition: 21-element numpy array for outflow gas composition.
    :param batch_location_history: List of batch locations.
    :param composition_history: List of batch compositions.
    """
    check_batch_and_composition_lengths(batch_location_history, composition_history)

    # If velocity is zero, return without changes
    if velocity == 0:
        return (
            batch_location_history,
            composition_history,
            outflow_composition
        )

    # Convert to NumPy arrays for vectorized calculations
    batch_location_history = np.array(batch_location_history, dtype=float)

    # Determine flow direction and absolute velocity
    flow_direction = 1 if velocity >= 0 else -1
    abs_velocity = abs(velocity)

    # Update batch locations
    batch_location_history += flow_direction * abs_velocity * time_step

    # Convert back to list for dynamic operations
    batch_location_history = batch_location_history.tolist()
    composition_history = [row for row in composition_history]

    # new_batch_locations = []
    # new_compositions = []
    #
    # for loc, comp in zip(batch_location_history, composition_history):
    #     if 0 <= loc <= length:
    #         new_batch_locations.append(loc)
    #         new_compositions.append(comp)
    #     elif loc > length and flow_direction == 1:
    #         outlet_composition = comp
    #     elif loc < 0 and flow_direction == -1:
    #         inlet_composition = comp
    #
    # batch_location_history = new_batch_locations
    # composition_history = new_compositions

    # Handle new batch based on flow direction
    if flow_direction == 1:  # Forward flow
        batch_location_history.append(0)
        composition_history.append(inflow_composition)
    else:  # Reverse flow
        batch_location_history.insert(0, length)
        composition_history.insert(0, inflow_composition)

    # Clean boundary batches
    (
        batch_location_history,
        composition_history,
        outflow_composition
    ) = clean_boundary_batches(
        outflow_composition,
        batch_location_history,
        composition_history,
        length,
        flow_direction,
    )

    return (
        batch_location_history,
        composition_history,
        outflow_composition
    )


def clean_boundary_batches(
        outflow_composition,
        batch_location_history,
        composition_history,
        length,
        flow_direction,
):
    """
    Cleans up batches at the boundaries of the pipeline.

    :param outflow_composition: Composition of the gas outflow
    :param batch_location_history: List of batch locations.
    :param composition_history: List of batch compositions.
    :param length: Length of the pipeline [m].
    :param flow_direction: Current flow direction (+1 for forward, -1 for reverse).
    :return: Updated batch_location_history, composition_history, inlet_composition, outlet_composition.
    """
    if flow_direction == 1:  # Forward flow
        while batch_location_history and batch_location_history[0] >= length:
            outflow_composition = composition_history.pop(0)
            batch_location_history.pop(0)
    else:  # Reverse flow
        while batch_location_history and batch_location_history[-1] <= 0:
            outflow_composition = composition_history.pop(-1)
            batch_location_history.pop(-1)

    return (
        batch_location_history,
        composition_history,
        outflow_composition
    )


def gas_composition_tracking(connection, time_step, method="simple_mixing"):
    """
    Function to track gas composition and corresponding batch head locations inside a pipeline
    :param connection:
    :param time_step: Time series resolution [s]
    :param method: Method to track gas composition
    :return:
    """
    composition_history = connection.composition_history
    batch_location_history = connection.batch_location_history
    length = connection.length
    velocity = connection.flow_velocity

    # Record inflow gas mixture composition
    if velocity is None:
        velocity = 0

    inlet_composition = connection.inlet.gas_mixture.eos_composition_tmp
    outlet_composition = connection.outlet.gas_mixture.eos_composition_tmp

    # if velocity >= 0:
    #     outflow_composition = outlet_composition
    # else:
    #     outflow_composition = inlet_composition

    inflow_composition = inlet_composition if velocity >= 0 else outlet_composition
    outflow_composition = outlet_composition if velocity >= 0 else inlet_composition

    if method == "batch_tracking":
        batch_location_history, composition_history, outflow_composition = batch_tracking(
            time_step, velocity, length, inflow_composition, outflow_composition, batch_location_history,
            composition_history
        )
        # outflow_composition = outlet_composition if velocity >= 0 else inlet_composition
        # if velocity >= 0:
        #     outflow_composition = connection.outlet.gas_mixture.eos_composition_tmp
        # else:
        #     outflow_composition = connection.inlet.gas_mixture.eos_composition_tmp
    elif method == "simple_mixing":
        outflow_composition = inflow_composition
    else:
        print(f"Method {method} not implemented yet!")

    # connection.outflow_composition = outflow_composition

    return batch_location_history, composition_history, outflow_composition


def create_incidence_matrix(nodes, connections):
    """Legacy function for backward compatibility."""
    branch_flow_matrix = create_branch_flow_matrix_legacy(nodes, connections)
    incidence_matrix = math.copysign(1, branch_flow_matrix)

    return incidence_matrix


def create_branch_flow_matrix(network, use_cuda=False):
    """
    Create branch flow matrix using network object.
    
    Args:
        network: Network object containing nodes and connections
        use_cuda: Whether to use CUDA
    """
    nodes = network.nodes
    connections = network.connections
    n_nodes = network.get_simulation_node_count()
    n_edges = len(connections)
    
    _branch_flow_matrix = np.zeros((n_edges, n_nodes))
    for _i, _connection in connections.items():
        inlet_sim_idx = network.node_id_to_simulation_node_index(_connection.inlet_index)
        outlet_sim_idx = network.node_id_to_simulation_node_index(_connection.outlet_index)
        _branch_flow_matrix[_i][inlet_sim_idx] = -_connection.flow_rate
        _branch_flow_matrix[_i][outlet_sim_idx] = _connection.flow_rate
        
    return _branch_flow_matrix


def create_branch_flow_matrix_legacy(nodes, connections, use_cuda=False):
    """
    Legacy version for backward compatibility.
    """
    n_nodes = len(nodes)
    n_edges = len(connections)
    
    _branch_flow_matrix = np.zeros((n_edges, n_nodes))
    for _i, _connection in connections.items():
        inlet_dense_idx = _connection.inlet_index - 1
        outlet_dense_idx = _connection.outlet_index - 1
        _branch_flow_matrix[_i][inlet_dense_idx] = -_connection.flow_rate
        _branch_flow_matrix[_i][outlet_dense_idx] = _connection.flow_rate
        
    return _branch_flow_matrix


def create_directed_graph_using_flow_directions(pipelines: dict):
    """
    Creates a directed graph from pipelines using their flow directions.

    :param pipelines: A dictionary of pipelines.

    :return: A tuple containing:
             - G: A NetworkX MultiDiGraph representing the pipelines with directed edges.
             - edge_index: A dictionary mapping edge identifiers to pipeline indices.
    """
    G = nx.MultiDiGraph()
    edge_index = {}
    for i, pipeline in pipelines.items():
        if pipeline.flow_velocity is None:
            G.add_edge(
                pipeline.inlet_index,
                pipeline.outlet_index,
                key=i,  # Use the pipeline index as the key
                flow_rate=1.0,
                composition=[],
            )
            edge_index[(pipeline.inlet_index, pipeline.outlet_index, i)] = i
        elif pipeline.flow_velocity >= 0:
            G.add_edge(pipeline.inlet_index, pipeline.outlet_index, key=i)
            edge_index[(pipeline.inlet_index, pipeline.outlet_index, i)] = i
        else:
            G.add_edge(pipeline.outlet_index, pipeline.inlet_index, key=i)
            edge_index[(pipeline.outlet_index, pipeline.inlet_index, i)] = i
    return G, edge_index


def topological_sort_of_nodes(graph: nx.MultiDiGraph):
    """
    Perform a topological sort of the nodes in a MultiDiGraph.

    :param graph: A NetworkX MultiDiGraph representing the DAG.
    :return: A list of nodes in topological order.
    """
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError(
            "The graph must be a Directed Acyclic Graph (DAG) to perform topological sorting."
        )

    nodes_topological_order = list(nx.topological_sort(graph))

    return nodes_topological_order


def topological_sort_of_edges(graph: nx.MultiDiGraph, edge_index: dict):
    """
    Perform a topological sort of the edges in a MultiDiGraph.

    :param graph: A MultiDiGraph representing the network.
    :param edge_index: A dictionary mapping (node, successor, key) to edge indices.
    :return: A list of edge indices in topological order.
    """
    edge_indices_order = []
    nodes_topological_order = topological_sort_of_nodes(graph)
    for node in nodes_topological_order:
        for successor in graph.successors(node):
            # Get all edges between node and successor
            edges = graph.get_edge_data(node, successor)
            for key in edges:
                edge_indices_order.append(edge_index[(node, successor, key)])
    return edge_indices_order


def create_nodal_composition_matrix(network, use_cuda=False):
    """
    Create nodal composition matrix using network object.
    
    Args:
        network: Network object containing nodes and connections
        use_cuda: Whether to use CUDA
    """
    _branch_flow_matrix = create_branch_flow_matrix(network, use_cuda=use_cuda)

    _nodal_inflow_matrix = np.where(_branch_flow_matrix > 0, _branch_flow_matrix, 0)

    _branch_outflow_composition = np.array(
        [c.outflow_composition for c in network.connections.values()]
    )
    _nodal_inflow_composition = np.dot(
        _nodal_inflow_matrix.T, _branch_outflow_composition
    )

    _nodal_inflow_vector = np.sum(
        np.where(_branch_flow_matrix > 0, _branch_flow_matrix, 0), axis=0
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        _nodal_composition_matrix = _nodal_inflow_composition.T / _nodal_inflow_vector
    # _nodal_composition_matrix = _nodal_inflow_composition.T / _nodal_inflow_vector

    return _nodal_composition_matrix


# @njit(boolean(float64[:, :], float64[:, :], float64, float64))
def allclose_with_nan(a, b, rtol=1e-03, atol=1e-04):
    # Check if arrays are close, considering NaNs
    nan_equal = np.isnan(a) & np.isnan(b)
    close_equal = np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=False)
    return np.all(nan_equal | close_equal)


def calculate_nodal_inflow_states(
        nodes,
        pipelines,
        cached_batch_information,
        connections,
        mapping_connections,
        tracking_method="simple_mixing",
        use_cuda=False,
        time_step=3600,
        network=None,
):
    to_update = True
    _prev_nodal_composition_matrix = np.zeros((21, (len(nodes))))

    # _count_nodal_inflow_iterations = 0
    pipelines = {i + 1: c for i, c in connections.items() if type(c) == Pipeline}
    graph, edge_index = create_directed_graph_using_flow_directions(connections)
    edge_orders = topological_sort_of_edges(graph, edge_index)

    while to_update:
        # connections_copy = connections.copy()
        # if tracking_method == "batch_tracking":
        #     # reset pipeline.batch_location_history and pipeline.composition_history
        #     for pipeline_id, pipeline in pipelines.items():
        #         batch_history, composition_history = cached_batch_information[pipeline_id]
        #         pipeline.batch_location_history = batch_history[:]
        #         pipeline.composition_history = composition_history[:]

        # _count_nodal_inflow_iterations += 1
        for connection_id in edge_orders:
            if isinstance(connections[connection_id], Pipeline):
                pipeline = connections[connection_id]
                if tracking_method == "batch_tracking":
                    batch_history, composition_history = cached_batch_information[pipeline.pipeline_index]
                    pipeline.batch_location_history = batch_history[:]
                    pipeline.composition_history = composition_history[:]
                    pipeline.inlet = nodes[pipeline.inlet_index]
                    pipeline.outlet = nodes[pipeline.outlet_index]

                (pipeline.batch_location_history,
                 pipeline.composition_history,
                 pipeline.outflow_composition,
                 ) = gas_composition_tracking(pipeline, time_step=time_step, method=tracking_method)

                if pipelines[pipeline.pipeline_index].outflow_composition is None:
                    raise ValueError("Check the topological order!")

        _nodal_composition_matrix = create_nodal_composition_matrix(network)

        nodes = update_temporary_nodal_gas_mixture_properties(
            network, _nodal_composition_matrix
        )
        if allclose_with_nan(_nodal_composition_matrix, _prev_nodal_composition_matrix):
            to_update = False
        else:
            _prev_nodal_composition_matrix = _nodal_composition_matrix

    # print(_count_nodal_inflow_iterations)

    return _nodal_composition_matrix, pipelines, nodes


def update_temporary_nodal_gas_mixture_properties(network, nodal_composition_matrix):
    """
    Update temporary nodal gas mixture properties using network object.
    
    Args:
        network: Network object containing nodes and index mapping
        nodal_composition_matrix: Matrix of nodal compositions
    """
    nodes = network.nodes
    for _i in range(nodal_composition_matrix.shape[1]):  # iterate over nodes
        if np.any(np.isnan(nodal_composition_matrix[:, _i])):  # No inflow
            pass
        else:
            node_id = network.simulation_node_index_to_node_id(_i)
            nodes[node_id].gas_mixture.eos_composition_tmp = nodal_composition_matrix[:, _i]
    return nodes


def calculate_flow_matrix(network, pressure_bar):
    connections = network.connections
    nodes = network.nodes
    n_nodes = len(nodes)
    flow_mat = np.zeros((n_nodes, n_nodes), dtype=float)

    pressure_index = 0
    for node in nodes.values():
        if node.index not in network.non_junction_nodes:
            node.pressure = pressure_bar[pressure_index] * 1e5
            pressure_index += 1

    for connection in connections.values():
        i = connection.inlet_index - 1
        j = connection.outlet_index - 1
        connection.inlet = nodes[i + 1]
        connection.outlet = nodes[j + 1]

        flow_direction = connection.determine_flow_direction()

        p1 = nodes[i + 1].pressure
        p2 = nodes[j + 1].pressure

        slope_correction = connection.calc_pipe_slope_correction()
        temp = connection.calculate_coefficient_for_iteration()

        flow_rate = (
                flow_direction * abs(p1 ** 2 - p2 ** 2 - slope_correction) ** (1 / 2) * temp
        )

        flow_mat[i][j] = -flow_rate
        flow_mat[j][i] = flow_rate

    return flow_mat


def calculate_flow_vector(network, pressure_bar, target_flow):
    flow_matrix = calculate_flow_matrix(network, pressure_bar)
    n_nodes = len(network.nodes.values())
    nodal_flow = np.dot(flow_matrix, np.ones(n_nodes))
    nodal_flow = [
        nodal_flow[i]
        for i in range(len(nodal_flow))
        if i + 1 not in network.non_junction_nodes
    ]
    delta_flow = target_flow - nodal_flow

    # delta_flow = [delta_flow[i] for i in range(len(delta_flow)) if i + 1 not in network.non_junction_nodes]
    return delta_flow


def plot_network_demand_distribution(network):
    nodes = network.nodes.values()
    node_demand = [n.volumetric_flow for n in nodes if n.volumetric_flow is not None]
    sns.histplot(data=node_demand, stat="probability")
    plt.xlim((min(node_demand) - 10, max(node_demand) + 10))
    plt.xlabel("Nodal volumetric flow demand [sm^3/s]")
    plt.show()
    return None


def check_square_matrix(a):
    return a.shape[0] == a.shape[1]


def check_symmetric(a, rtol=1e-05, atol=1e-08):
    return np.allclose(a, a.T, rtol=rtol, atol=atol)


def check_all_off_diagonal_elements(a, criterion):
    res = True

    if check_square_matrix(a):
        pass
    else:
        print("Matrix is not a square matrix!")

    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            if i != j:
                if criterion == "zero":
                    res = a[i][j] == 0
                elif criterion == "positive":
                    res = a[i][j] > 0
                elif criterion == "non-negative":
                    res = a[i][j] >= 0
                elif criterion == "negative":
                    res = a[i][j] < 0
                elif criterion == "non-positive":
                    res = a[i][j] <= 0
                else:
                    print("Check the given criterion!")
                    return False
                if res == False:
                    return False
    return res


def check_all_diagonal_elements(a, criterion):
    res = True

    if check_square_matrix(a):
        pass
    else:
        print("Matrix is not a square matrix!")

    if criterion == "zero":
        res = (np.diagonal(a) == 0).all()
    elif criterion == "positive":
        res = (np.diagonal(a) > 0).all()
    elif criterion == "non-negative":
        res = (np.diagonal(a) >= 0).all()
    elif criterion == "negative":
        res = (np.diagonal(a) < 0).all()
    elif criterion == "non-positive":
        res = (np.diagonal(a) <= 0).all()
    else:
        print("Check the given criterion!")
        return False

    return res
