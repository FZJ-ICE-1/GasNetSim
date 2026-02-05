#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
#   ******************************************************************************
#     Copyright (c) 2024.
#     Developed by Yifei Lu
#     Last change on 12/19/24, 10:24 AM
#     Last change by yifei
#    *****************************************************************************
from collections import OrderedDict
import math
from pathlib import Path
import numpy as np
import pandas as pd
import warnings
from scipy.constants import atm
from typing import Callable, Dict, Optional, Tuple

from ..network import Network
from ..node import Node
from ..pipeline import Pipeline, Resistance, ShortPipe, LinearResistance
from ..dynamic.dynamic_network import DynamicNetwork
from ..dynamic.dynamic_node import DynamicNode
from ..dynamic.dynamic_pipeline import DynamicPipeline
from ..compressor import Compressor
from ..gas_mixture.typical_mixture_composition import COMMON_GAS_COMPOSITIONS
from ...utils.exception import *

AVAILABLE_GAS_NAMES = ", ".join(COMMON_GAS_COMPOSITIONS.keys())

def get_builtin_gas_composition(name):
    if name not in COMMON_GAS_COMPOSITIONS:
        raise ValueError(
            f"The gas mixture '{name}' is not implemented.\n"
            f"Available gas mixtures: {AVAILABLE_GAS_NAMES}\n"
            f"If you want to use a specific gas mixture composition, please assign it as a OrderedDict or implement it separately."
        )
    return COMMON_GAS_COMPOSITIONS[name]


def convert_gas_composition(gas_composition: str) -> OrderedDict:
    try:
        return OrderedDict(eval(gas_composition))
    except (SyntaxError, ValueError):
        raise ValueError(
            f"Invalid gas composition provided: {gas_composition}."
            f"Please provide a valid built-in name or a correctly formatted dictionary string."
        )


def read_nodes(
    path_to_file: Path, base_composition=None, node_cls=Node
) -> dict[int, Node]:
    """
    Read nodes from a CSV file and create Node objects.

    :param path_to_file: Path to the CSV file containing nodes information.
    :return: A dictionary of node indices to Node objects.
    """
    nodes = {}
    df_node = pd.read_csv(path_to_file, delimiter=";")
    df_node = df_node.replace({np.nan: None})

    if base_composition is None:
        base_composition = COMMON_GAS_COMPOSITIONS["NATURAL_GAS_gri30"]

    for _, row in df_node.iterrows():
        # Convert gauge pressure to absolute pressure
        gauge_pressure_pa = row["pressure_pa"]
        absolute_pressure_pa = gauge_pressure_pa + atm if gauge_pressure_pa is not None else None

        if row["gas_composition"] is not None:
            gas_composition_str = row["gas_composition"]

            if "{" not in gas_composition_str and "}" not in gas_composition_str:
                # Assume it's a built-in gas composition name
                row["gas_composition"] = get_builtin_gas_composition(gas_composition_str)
            else:
                # Assume it's a string representation of a dictionary of gas composition
                row["gas_composition"] = convert_gas_composition(row["gas_composition"])
        else:
            row["gas_composition"] = base_composition

        nodes[row["node_index"]] = node_cls(
            node_index=row["node_index"],
            pressure_pa=absolute_pressure_pa,
            volumetric_flow=row["flow_sm3_per_s"],
            energy_flow=row["flow_MW"],
            temperature=row["temperature_k"],
            altitude=row["altitude_m"],
            gas_composition=row["gas_composition"],
            node_type=row["node_type"],
            flow_type=row["flow_type"],
            longitude=row.get("longitude"),
            latitude=row.get("latitude"),
        )
    return nodes


def read_pipelines(
    path_to_file: Path, network_nodes: dict, conversion_factor=1.0, pipe_cls=Pipeline
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
        friction_method = row.get("friction_method", "chen") or "chen"
        pipelines[row["pipeline_index"]] = pipe_cls(
            pipeline_index=row["pipeline_index"],
            inlet=network_nodes[row["inlet_index"]],
            outlet=network_nodes[row["outlet_index"]],
            diameter=row["diameter_m"],
            length=row["length_m"],
            friction_factor_method=friction_method,
            conversion_factor=conversion_factor,
        )
    return pipelines


def _apply_length_overrides(pipelines: dict, length_overrides: Optional[Dict[int, float]]):
    if not length_overrides:
        return
    for pipe_id, new_len in length_overrides.items():
        if pipe_id not in pipelines:
            raise ValueError(f"Pipeline ID {pipe_id} not found for length override.")
        length_val = float(new_len)
        if length_val <= 0:
            raise ValueError(f"Invalid length override for pipeline {pipe_id}: {new_len}")
        pipelines[pipe_id].length = length_val


def _default_interpolate_value(
    v0: Optional[float], v1: Optional[float], frac: float
) -> Optional[float]:
    if v0 is None and v1 is None:
        return None
    if v0 is None:
        return v1
    if v1 is None:
        return v0
    return v0 + (v1 - v0) * frac


def _segment_pipelines(
    nodes: dict,
    pipelines: dict,
    node_cls=Node,
    pipe_cls=Pipeline,
    segment_length_m: Optional[float] = None,
    segments_per_pipe: Optional[int] = None,
    temperature_interpolator: Optional[
        Callable[[Optional[float], Optional[float], float], Optional[float]]
    ] = None,
) -> Tuple[dict, dict]:
    """
    Discretize each pipeline into multiple segments by inserting intermediate nodes.
    Original (real) nodes keep their indices; new nodes are appended after max ID.
    temperature_interpolator: optional callable for temperature interpolation between endpoints.
    """
    if not pipelines:
        return nodes, pipelines

    if segments_per_pipe is None and segment_length_m is None:
        raise ValueError("Segmentation requested but no segment length/count provided.")
    if segments_per_pipe is not None:
        if int(segments_per_pipe) < 1:
            raise ValueError("segments_per_pipe must be >= 1.")
    if segment_length_m is not None:
        if float(segment_length_m) <= 0:
            raise ValueError("segment_length_m must be > 0.")

    new_nodes = dict(nodes)
    new_pipelines = dict()

    next_node_id = max(new_nodes.keys()) + 1 if new_nodes else 1
    # When segmenting, reindex pipelines sequentially (1..N) to preserve
    # assumptions in steady-state composition tracking.
    next_pipe_id = 1

    for pipe in pipelines.values():
        parent_id = pipe.pipeline_index
        L = float(pipe.length)
        if L <= 0:
            raise ValueError(f"Pipeline {pipe.pipeline_index} has invalid length {pipe.length}")

        if segments_per_pipe is not None:
            n_segments = int(segments_per_pipe)
        else:
            n_segments = int(math.ceil(L / float(segment_length_m)))
        n_segments = max(1, n_segments)

        if n_segments == 1:
            # Keep pipe as-is but reindex sequentially for segmented mode
            pipe.pipeline_index = next_pipe_id
            setattr(pipe, "parent_pipeline_index", parent_id)
            setattr(pipe, "segment_index", 0)
            new_pipelines[next_pipe_id] = pipe
            next_pipe_id += 1
            continue

        seg_length = L / n_segments

        inlet = pipe.inlet
        outlet = pipe.outlet

        h0 = getattr(inlet, "altitude", 0.0) or 0.0
        h1 = getattr(outlet, "altitude", 0.0) or 0.0
        t0 = getattr(inlet, "temperature", None)
        t1 = getattr(outlet, "temperature", None)
        gas_comp = getattr(inlet, "gas_composition", None) or getattr(outlet, "gas_composition", None)
        temp_interp = temperature_interpolator or _default_interpolate_value

        prev_node = inlet

        for seg_idx in range(1, n_segments):
            frac = seg_idx / n_segments
            altitude = h0 + (h1 - h0) * frac
            temp = temp_interp(t0, t1, frac)

            new_node = node_cls(
                node_index=next_node_id,
                pressure_pa=None,
                volumetric_flow=0.0,
                energy_flow=None,
                temperature=temp if temp is not None else 288.15,
                altitude=altitude,
                gas_composition=gas_comp,
                node_type="junction",
                flow_type="volumetric",
                longitude=None,
                latitude=None,
            )
            new_nodes[next_node_id] = new_node
            next_node_id += 1

            seg_pipe = pipe_cls(
                pipeline_index=next_pipe_id,
                inlet=prev_node,
                outlet=new_node,
                diameter=pipe.diameter,
                length=seg_length,
                efficiency=getattr(pipe, "efficiency", 0.85),
                roughness=getattr(pipe, "roughness", 0.000015),
                ambient_temp=getattr(pipe, "ambient_temp", 288.15),
                ambient_pressure=getattr(pipe, "ambient_pressure", 101325.0),
                heat_transfer_coefficient=getattr(pipe, "heat_transfer_coefficient", 3.69),
                valve=getattr(pipe, "valve", 0),
                friction_factor_method=getattr(pipe, "friction_factor_method", "chen"),
                conversion_factor=getattr(pipe, "conversion_factor", 1.0),
                constant_friction_factor=getattr(pipe, "constant_friction_factor", None),
            )
            setattr(seg_pipe, "parent_pipeline_index", parent_id)
            setattr(seg_pipe, "segment_index", seg_idx - 1)
            new_pipelines[next_pipe_id] = seg_pipe
            next_pipe_id += 1

            prev_node = new_node

        # Final segment to original outlet
        seg_pipe = pipe_cls(
            pipeline_index=next_pipe_id,
            inlet=prev_node,
            outlet=outlet,
            diameter=pipe.diameter,
            length=seg_length,
            efficiency=getattr(pipe, "efficiency", 0.85),
            roughness=getattr(pipe, "roughness", 0.000015),
            ambient_temp=getattr(pipe, "ambient_temp", 288.15),
            ambient_pressure=getattr(pipe, "ambient_pressure", 101325.0),
            heat_transfer_coefficient=getattr(pipe, "heat_transfer_coefficient", 3.69),
            valve=getattr(pipe, "valve", 0),
            friction_factor_method=getattr(pipe, "friction_factor_method", "chen"),
            conversion_factor=getattr(pipe, "conversion_factor", 1.0),
            constant_friction_factor=getattr(pipe, "constant_friction_factor", None),
        )
        setattr(seg_pipe, "parent_pipeline_index", parent_id)
        setattr(seg_pipe, "segment_index", n_segments - 1)
        new_pipelines[next_pipe_id] = seg_pipe
        next_pipe_id += 1

    return new_nodes, new_pipelines


def read_compressors(path_to_file: Path, network_nodes: dict) -> dict:
    """
    Read compressors from a CSV file and create Compressor objects.

    :param path_to_file: Path to the CSV file containing compressor information.
    :param network_nodes: Dictionary of existing network nodes.
    :return: A dictionary of compressor indices to Compressor objects.
    """
    compressors = dict()
    
    try:
        df_compressors = pd.read_csv(path_to_file, delimiter=";")
        df_compressors = df_compressors.replace({np.nan: None})
        
        for index, row in df_compressors.iterrows():
            compressor_index = int(row["compressor_index"])
            inlet_index = int(row["inlet"])
            outlet_index = int(row["outlet"])
            compression_ratio = float(row["compression_ratio"]) if row["compression_ratio"] is not None else 1.1
            efficiency = float(row["efficiency"]) if row["efficiency"] is not None else 0.85
            thermodynamic_process = row["thermodynamic_process"] if row["thermodynamic_process"] is not None else "isentropic"
            drive = row["drive"] if "drive" in row and row["drive"] is not None else "electric"
            
            # Get inlet and outlet nodes
            inlet_node = network_nodes[inlet_index]
            outlet_node = network_nodes[outlet_index]
            
            # Create compressor object
            compressor = Compressor(
                compressor_index=compressor_index,
                inlet=inlet_node,
                outlet=outlet_node,
                compression_ratio=compression_ratio,
                efficiency=efficiency,
                thermodynamic_process=thermodynamic_process,
                drive=drive
            )
            
            compressors[compressor_index] = compressor
            
    except FileNotFoundError:
        print(f"Compressor file not found: {path_to_file}")
    except Exception as e:
        print(f"Error reading compressors: {e}")
        
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


def create_network_from_csv(
    path_to_folder: Path,
    conversion_factor=1.0,
    base_composition=None,
    dynamic=False,
    segment_pipes: bool = False,
    segment_length_m: Optional[float] = None,
    segments_per_pipe: Optional[int] = None,
    length_overrides: Optional[Dict[int, float]] = None,
    temperature_interpolator: Optional[
        Callable[[Optional[float], Optional[float], float], Optional[float]]
    ] = None,
) -> Network:
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
    return create_network_from_folder(
        path_to_folder,
        conversion_factor,
        base_composition,
        dynamic,
        segment_pipes=segment_pipes,
        segment_length_m=segment_length_m,
        segments_per_pipe=segments_per_pipe,
        length_overrides=length_overrides,
        temperature_interpolator=temperature_interpolator,
    )


def create_network_from_folder(
    path_to_folder: Path,
    conversion_factor=1.0,
    base_composition=None,
    dynamic=False,
    segment_pipes: bool = False,
    segment_length_m: Optional[float] = None,
    segments_per_pipe: Optional[int] = None,
    length_overrides: Optional[Dict[int, float]] = None,
    temperature_interpolator: Optional[
        Callable[[Optional[float], Optional[float], float], Optional[float]]
    ] = None,
) -> Network:
    """
    Create a Network object from CSV files located in the specified folder.

    :param path_to_folder: Path to the folder containing the CSV files.
    :param conversion_factor: Conversion factor for pipeline data.
    :param segment_pipes: If True, split each pipeline into multiple segments.
    :param segment_length_m: Target segment length [m] when segmenting.
    :param segments_per_pipe: Fixed number of segments per pipe (overrides segment_length_m).
    :param length_overrides: Optional dict {pipeline_id: length_m} to override CSV lengths.
    :param temperature_interpolator: Optional callable to interpolate temperatures along segmented pipes.
    :return: A Network object.
    """
    all_files = list(path_to_folder.glob("*.csv"))
    nodes_file = next((file for file in all_files if "node" in file.stem), None)

    if nodes_file is None:
        raise FileNotFoundError("Nodes file is required to create the network.")

    node_cls = DynamicNode if dynamic else Node
    pipe_cls = DynamicPipeline if dynamic else Pipeline
    net_cls = DynamicNetwork if dynamic else Network

    nodes = read_nodes(
        nodes_file, base_composition=base_composition, node_cls=node_cls
    )

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
                        file, nodes, conversion_factor, pipe_cls=pipe_cls
                    )
                else:
                    network_components[component_key + "s"] = read_function(file, nodes)
                break

    # Optional: override pipe lengths before segmentation
    if network_components["pipelines"] is not None and length_overrides:
        _apply_length_overrides(network_components["pipelines"], length_overrides)

    # Optional: segment pipelines (insert intermediate nodes)
    if segment_pipes and network_components["pipelines"] is not None:
        nodes, pipelines = _segment_pipelines(
            nodes=network_components["nodes"],
            pipelines=network_components["pipelines"],
            node_cls=node_cls,
            pipe_cls=pipe_cls,
            segment_length_m=segment_length_m,
            segments_per_pipe=segments_per_pipe,
            temperature_interpolator=temperature_interpolator,
        )
        network_components["nodes"] = nodes
        network_components["pipelines"] = pipelines

    # Create and return the Network object
    return net_cls(
        nodes=network_components["nodes"],
        pipelines=network_components["pipelines"],
        compressors=network_components["compressors"],
        resistances=network_components["resistances"],
        linear_resistances=network_components["linear_resistances"],
        shortpipes=network_components["shortpipes"],
    )


def create_network_from_files(
    component_files: dict[str, Path],
    conversion_factor=1.0,
    base_composition=None,
    dynamic=False,
    segment_pipes: bool = False,
    segment_length_m: Optional[float] = None,
    segments_per_pipe: Optional[int] = None,
    length_overrides: Optional[Dict[int, float]] = None,
    temperature_interpolator: Optional[
        Callable[[Optional[float], Optional[float], float], Optional[float]]
    ] = None,
) -> Network:
    """
    Create a Network object from specified component CSV files.

    :param component_files: A dictionary mapping component names (e.g., 'nodes', 'pipelines') to file paths.
    :param conversion_factor: Conversion factor for pipeline data.
    :param segment_pipes: If True, split each pipeline into multiple segments.
    :param segment_length_m: Target segment length [m] when segmenting.
    :param segments_per_pipe: Fixed number of segments per pipe (overrides segment_length_m).
    :param length_overrides: Optional dict {pipeline_id: length_m} to override CSV lengths.
    :param temperature_interpolator: Optional callable to interpolate temperatures along segmented pipes.
    :return: A Network object.
    """
    # Ensure nodes file is provided
    nodes_file = component_files.get("nodes")
    if nodes_file is None:
        raise ValueError("Nodes file is required to create the network.")

    # Read nodes
    node_cls = DynamicNode if dynamic else Node
    pipe_cls = DynamicPipeline if dynamic else Pipeline
    net_cls = DynamicNetwork if dynamic else Network

    nodes = read_nodes(nodes_file, base_composition=base_composition, node_cls=node_cls)

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
        "pipelines": read_pipelines,
        "compressors": read_compressors,
        "resistances": read_resistances,
        "linear_resistances": read_linear_resistances,
        "shortpipes": read_shortpipes,
    }

    # Read other components if provided
    for component_name, read_function in read_functions.items():
        if component_name in component_files:
            if component_name == "pipelines":
                network_components[component_name] = read_function(
                    component_files[component_name],
                    nodes,
                    conversion_factor,
                    pipe_cls=pipe_cls,
                )
            else:
                network_components[component_name] = read_function(
                    component_files[component_name], nodes
                )

    # Optional: override pipe lengths before segmentation
    if network_components["pipelines"] is not None and length_overrides:
        _apply_length_overrides(network_components["pipelines"], length_overrides)

    # Optional: segment pipelines (insert intermediate nodes)
    if segment_pipes and network_components["pipelines"] is not None:
        nodes, pipelines = _segment_pipelines(
            nodes=network_components["nodes"],
            pipelines=network_components["pipelines"],
            node_cls=node_cls,
            pipe_cls=pipe_cls,
            segment_length_m=segment_length_m,
            segments_per_pipe=segments_per_pipe,
            temperature_interpolator=temperature_interpolator,
        )
        network_components["nodes"] = nodes
        network_components["pipelines"] = pipelines

    # Create and return the Network object
    return net_cls(
        nodes=network_components["nodes"],
        pipelines=network_components["pipelines"],
        compressors=network_components["compressors"],
        resistances=network_components["resistances"],
        linear_resistances=network_components["linear_resistances"],
        shortpipes=network_components["shortpipes"],
    )
