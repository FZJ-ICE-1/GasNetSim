#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
#   ******************************************************************************
#     Copyright (c) 2024.
#     Developed by Yifei Lu
#     Last change on 10/22/24, 12:35 PM
#     Last change by yifei
#    *****************************************************************************
from collections import OrderedDict
from pathlib import Path
import numpy as np
import pandas as pd
import warnings

from ..network import Network
from ..node import Node
from ..pipeline import Pipeline, Resistance, ShortPipe, LinearResistance
from ...utils.exception import *


def read_nodes(path_to_file: Path) -> dict[int, Node]:
    """
    Read nodes from a CSV file and create Node objects.

    :param path_to_file: Path to the CSV file containing nodes information.
    :return: A dictionary of node indices to Node objects.
    """
    nodes = {}
    df_node = pd.read_csv(path_to_file, delimiter=";")
    df_node = df_node.replace({np.nan: None})

    for _, row in df_node.iterrows():
        if row["gas_composition"] is not None:
            row["gas_composition"] = OrderedDict(eval(row["gas_composition"]))

        nodes[row["node_index"]] = Node(
            node_index=row["node_index"],
            pressure_pa=row["pressure_pa"],
            volumetric_flow=row["flow_sm3_per_s"],
            energy_flow=row["flow_MW"],
            temperature=row["temperature_k"],
            altitude=row["altitude_m"],
            gas_composition=row["gas_composition"],
            node_type=row["node_type"],
            longitude=row.get("longitude"),
            latitude=row.get("latitude"),
        )
    return nodes


def read_pipelines(
    path_to_file: Path, network_nodes: dict, conversion_factor=1.0
) -> dict:
    """

    :param path_to_file:
    :param network_nodes:
    :return:
    """
    pipelines = dict()
    df_pipe = pd.read_csv(path_to_file, delimiter=";")
    df_pipe = df_pipe.replace({np.nan: None})

    for row_index, row in df_pipe.iterrows():
        pipelines[row["pipeline_index"]] = Pipeline(
            inlet=network_nodes[row["inlet_index"]],
            outlet=network_nodes[row["outlet_index"]],
            diameter=row["diameter_m"],
            length=row["length_m"],
            friction_factor_method=row["friction_method"],
            conversion_factor=conversion_factor,
        )
    return pipelines


def read_compressors(path_to_file: Path) -> dict:
    """

    :param path_to_file:
    :return:
    """
    compressors = dict()
    return compressors


def read_resistances(path_to_file: Path, network_nodes: dict) -> dict:
    """

    :param path_to_file:
    :param network_nodes:
    :return:
    """
    resistances = dict()
    df_resistance = pd.read_csv(path_to_file, delimiter=";")
    df_resistance = df_resistance.replace({np.nan: None})

    for row_index, row in df_resistance.iterrows():
        resistances[row["resistance_index"]] = Resistance(
            inlet=network_nodes[row["inlet_index"]],
            outlet=network_nodes[row["outlet_index"]],
            resistance=row["resistance"],
        )
    return resistances


def read_linear_resistances(path_to_file: Path, network_nodes: dict) -> dict:
    """

    :param path_to_file:
    :param network_nodes:
    :return:
    """
    resistances = dict()
    df_linear_resistance = pd.read_csv(path_to_file, delimiter=";")
    df_linear_resistance = df_linear_resistance.replace({np.nan: None})

    for row_index, row in df_linear_resistance.iterrows():
        resistances[row["linear_resistance_index"]] = LinearResistance(
            inlet=network_nodes[row["inlet_index"]],
            outlet=network_nodes[row["outlet_index"]],
            resistance=row["linear_resistance"],
        )
    return resistances


def read_shortpipes(path_to_file: Path, network_nodes: dict) -> dict:
    """

    :param path_to_file:
    :param network_nodes:
    :return:
    """
    shortpipes = dict()
    df_shortpipes = pd.read_csv(path_to_file, delimiter=";")
    df_shortpipes = df_shortpipes.replace({np.nan: None})

    for row_index, row in df_shortpipes.iterrows():
        shortpipes[row["shortpipe_index"]] = ShortPipe(
            inlet=network_nodes[row["inlet_index"]],
            outlet=network_nodes[row["outlet_index"]],
        )
    return shortpipes


import warnings


def create_network_from_csv(path_to_folder: Path, conversion_factor=1.0) -> Network:
    """
    Create a Network object from CSV files located in the specified folder.

    :param path_to_folder: Path to the folder containing the CSV files.
    :param conversion_factor: Conversion factor for pipeline data.
    :return: A Network object.
    """
    warnings.warn(
        "create_network_from_csv() is deprecated and will be removed in a future version. "
        "Please use create_network_from_folder() instead.",
        FutureWarning,
        stacklevel=2,
    )
    return create_network_from_folder(path_to_folder, conversion_factor)


def create_network_from_folder(path_to_folder: Path, conversion_factor=1.0) -> Network:
    """
    Create a Network object from CSV files located in the specified folder.

    :param path_to_folder: Path to the folder containing the CSV files.
    :param conversion_factor: Conversion factor for pipeline data.
    :return: A Network object.
    """
    all_files = list(path_to_folder.glob("*.csv"))
    nodes_file = next((file for file in all_files if "node" in file.stem), None)

    if nodes_file is None:
        raise FileNotFoundError("Nodes file is required to create the network.")

    nodes = read_nodes(nodes_file)

    # Initialize network components
    network_components = {
        "nodes": nodes,
        "pipelines": None,
        "compressors": None,
        "resistances": None,
        "shortpipes": None,
        "linear_resistances": None,
    }

    # Mapping of component names to their corresponding read functions
    read_functions = {
        "pipeline": read_pipelines,
        "compressor": read_compressors,
        "resistance": read_resistances,
        "linearR": read_linear_resistances,
        "shortpipe": read_shortpipes,
    }

    # Read other components if provided
    for file in all_files:
        file_name = file.stem
        for component_key, read_function in read_functions.items():
            if component_key in file_name:
                if component_key == "pipeline":
                    network_components[component_key + "s"] = read_function(
                        file, nodes, conversion_factor
                    )
                else:
                    network_components[component_key + "s"] = read_function(file, nodes)
                break

    # Create and return the Network object
    return Network(
        nodes=network_components["nodes"],
        pipelines=network_components["pipelines"],
        compressors=network_components["compressors"],
        resistances=network_components["resistances"],
        linear_resistances=network_components["linear_resistances"],
        shortpipes=network_components["shortpipes"],
    )
