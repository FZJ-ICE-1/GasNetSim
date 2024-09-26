#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
#   ******************************************************************************
#     Copyright (c) 2024.
#     Developed by Yifei Lu
#     Last change on 9/18/24, 10:57 PM
#     Last change by yifei
#    *****************************************************************************
import copy
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
    outflow_composition = connection.outflow_composition

    # Record inflow gas mixture composition
    if velocity is None:
        velocity = 0
    if velocity >= 0:
        inflow_composition = connection.inlet.gas_mixture.eos_composition_tmp
    else:
        inflow_composition = connection.outlet.gas_mixture.eos_composition_tmp

    if method == "batch_tracking":
        # Batch-tracking
        batch_location_history += (
            time_step * velocity
        )  # Update batch head compositions and locations
        batch_location_history = np.append(batch_location_history, 0)
        composition_history = np.append(composition_history, inflow_composition)

        # Update outflow composition
        while (
            abs(batch_location_history[0]) >= length
        ):  # if the head of a batch reached the end of the pipeline
            outflow_composition = composition_history[0]
            composition_history, batch_location_history = (
                composition_history[1:],
                batch_location_history[1:],
            )

        # update connection composition and batch location history
        connection.composition_history = composition_history
        connection.batch_location_history = batch_location_history
    elif method == "simple_mixing":
        outflow_composition = copy.deepcopy(inflow_composition)
    else:
        print(f"Method {method} not implemented yet!")

    connection.outflow_composition = outflow_composition  # Update outflow composition
    return connection


def create_incidence_matrix(nodes, connections):
    branch_flow_matrix = create_branch_flow_matrix(nodes, connections)
    incidence_matrix = math.copysign(1, branch_flow_matrix)

    return incidence_matrix


def create_branch_flow_matrix(nodes, connections, use_cuda=False):
    n_nodes = len(nodes)
    n_edges = len(connections)

    _branch_flow_matrix = np.zeros((n_edges, n_nodes))
    for _i, _connection in connections.items():
        _branch_flow_matrix[_i][_connection.inlet_index - 1] = -_connection.flow_rate
        _branch_flow_matrix[_i][_connection.outlet_index - 1] = _connection.flow_rate
    return _branch_flow_matrix


def create_directed_graph_using_flow_directions(pipelines):
    G = nx.DiGraph()
    edge_index = {}
    for i, pipeline in pipelines.items():
        if pipeline.flow_velocity is None:
            G.add_edge(
                pipeline.inlet_index,
                pipeline.outlet_index,
                flow_rate=1.0,
                composition=[],
            )
            edge_index[(pipeline.inlet_index, pipeline.outlet_index)] = i
        elif pipeline.flow_velocity >= 0:
            G.add_edge(pipeline.inlet_index, pipeline.outlet_index)
            edge_index[(pipeline.inlet_index, pipeline.outlet_index)] = i
        else:
            G.add_edge(pipeline.outlet_index, pipeline.inlet_index)
            edge_index[(pipeline.outlet_index, pipeline.inlet_index)] = i
    return G, edge_index


def topological_sort_of_nodes(graph):
    if not nx.is_directed_acyclic_graph(graph):
        raise ValueError(
            "The graph must be a Directed Acyclic Graph (DAG) to perform topological sorting."
        )

    nodes_topological_order = list(nx.topological_sort(graph))

    return nodes_topological_order


def topological_sort_of_edges(graph, edge_index):
    edge_indices_order = []
    nodes_topological_order = topological_sort_of_nodes(graph)
    for node in nodes_topological_order:
        for successor in graph.successors(node):
            edge_indices_order.append(edge_index[(node, successor)])
    return edge_indices_order


def create_nodal_composition_matrix(nodes, connections, use_cuda=False):
    _branch_flow_matrix = create_branch_flow_matrix(nodes, connections)

    _nodal_inflow_matrix = np.where(_branch_flow_matrix > 0, _branch_flow_matrix, 0)

    _branch_outflow_composition = np.array(
        [c.outflow_composition for c in connections.values()]
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
    connections,
    mapping_connections,
    tracking_method="simple_mixing",
    use_cuda=False,
    time_step=0,
):
    to_update = True
    _prev_nodal_composition_matrix = np.zeros((21, (len(nodes))))

    # _count_nodal_inflow_iterations = 0

    pipelines = {i: c for i, c in connections.items() if type(c) == Pipeline}
    graph, edge_index = create_directed_graph_using_flow_directions(pipelines)
    edge_orders = topological_sort_of_edges(graph, edge_index)

    while to_update:
        # _count_nodal_inflow_iterations += 1
        for i in edge_orders:
            pipelines[i] = gas_composition_tracking(
                pipelines[i], time_step=time_step, method=tracking_method
            )

        _nodal_composition_matrix = create_nodal_composition_matrix(nodes, connections)

        nodes = update_temporary_nodal_gas_mixture_properties(
            nodes, _nodal_composition_matrix
        )
        if allclose_with_nan(_nodal_composition_matrix, _prev_nodal_composition_matrix):
            to_update = False
        else:
            _prev_nodal_composition_matrix = _nodal_composition_matrix

    # print(_count_nodal_inflow_iterations)

    return _nodal_composition_matrix


def update_temporary_nodal_gas_mixture_properties(nodes, nodal_composition_matrix):
    """

    :param nodes:
    :param nodal_composition_matrix:
    :return:
    """
    for _i in range(nodal_composition_matrix.shape[1]):  # iterate over nodes
        if np.any(np.isnan(nodal_composition_matrix[:, _i])):  # No inflow
            pass
        else:
            nodes[_i + 1].gas_mixture.eos_composition_tmp = nodal_composition_matrix[
                :, _i
            ]
            # nodes[_i+1].gas_mixture.update_gas_mixture()
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
            flow_direction * abs(p1**2 - p2**2 - slope_correction) ** (1 / 2) * temp
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
