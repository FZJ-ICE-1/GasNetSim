#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
#   ******************************************************************************
#     Copyright (c) 2024.
#     Developed by Yifei Lu
#     Last change on 12/25/24, 11:16 PM
#     Last change by yifei
#    *****************************************************************************
import pandas as pd
from pathlib import Path
import logging
import copy
from tqdm import tqdm

from ..components.network import Network
from ..components.utils.utils import plot_network_demand_distribution
from ..components.utils.cuda_support import *


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARNING)


VALID_RESULT_KEYS = [
    "nodal_pressure",
    "pipeline_flowrate",
    "nodal_gas_composition",
    "nodal_HHV_MJ_per_sm3",
    "nodal_WI_MJ_per_sm3",
    "nodal_volume_flow_sm3_per_s",
    "nodal_energy_flow_MW",
]


def read_profiles(file, sep=";"):
    profiles = pd.read_csv(Path(file), sep=sep)
    logger.info(f"Reading profiles from {file}.")
    return profiles


def print_progress_bar(
    iteration, total, prefix="", suffix="", decimals=1, length=100, fill="█"
):
    """
    Call in a loop to create terminal progress bar.
    the code is mentioned in : https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filled_length = int(length * iteration // total)
    bar = fill * filled_length + "-" * (length - filled_length)
    # logger.info('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end="")
    # Print New Line on Complete
    if iteration == total:
        print("\n")


def check_profiles(profiles):
    if profiles["time"].dtype == int:
        print()
    elif profiles["time"].dtype == pd.Timestamp:
        print()
        # df['time'] = df['time'].apply(lambda x: pd.Timestamp.now() + pd.Timedelta(seconds=x))


def run_snapshot(network, tol=0.01, use_cuda=False, tracking_method="simple_mixing"):
    # plot_network_demand_distribution(network)
    if use_cuda:
        is_cuda_available()
    network = network.simulation(
        tol=tol, use_cuda=use_cuda, tracking_method=tracking_method
    )
    return network


def not_converged(time_step, ts_variables):
    logger.error(f"CalculationNotConverged at time step {time_step}.")
    if not ts_variables["continue_on_divergence"]:
        raise ts_variables["errors"][0]


def update_network_topology(network):
    full_network = copy.deepcopy(network)  # make a copy of the fully connected network
    network_nodes = full_network.nodes
    network_pipes = full_network.pipelines
    network_resistances = full_network.resistances

    # Check which nodes need to be removed
    removed_nodes = dict()
    remaining_nodes = dict()
    for i, node in list(network_nodes.items()):
        if node.flow == 0:
            removed_nodes[i] = node
        else:
            remaining_nodes[i - len(removed_nodes)] = node

    # Check which pipelines need to be removed
    removed_pipes = dict()
    remaining_pipes = dict()
    for i, pipe in list(network_resistances.items()):
        if (pipe.inlet_index in removed_nodes.keys()) or (
            pipe.outlet_index in removed_nodes.keys()
        ):
            pipe.valve = 1
        else:
            pipe.valve = 0
        if pipe.valve == 1:
            removed_pipes[i] = pipe
        else:
            pipe.inlet_index = list(remaining_nodes.keys())[
                list(remaining_nodes.values()).index(pipe.inlet)
            ]
            pipe.outlet_index = list(remaining_nodes.keys())[
                list(remaining_nodes.values()).index(pipe.outlet)
            ]
            remaining_pipes[i - len(removed_pipes)] = pipe

    return Network(nodes=remaining_nodes, pipelines=None, resistances=remaining_pipes)


def run_time_series(
    network,
    file=None,
    sep=";",
    profile_type="energy",
    composition_tracking=False,
    output_format="excel",
    output_filename="time_series_results",
    results_to_save=["nodal_pressure", "pipeline_flowrate", "nodal_gas_composition"],
):
    """
    Run time series simulation for the network and save results in specified format.
    """
    # Validate results_to_save before running the simulation
    validate_results_to_save(results_to_save)

    # Initialize
    full_network = copy.deepcopy(network)
    results = dict([(k, []) for k in results_to_save])

    # Read profiles
    if file is not None:
        profiles = read_profiles(file, sep=sep)
        time_steps = profiles.index
    else:
        time_steps = range(5)  # Test with 5 fictitious time steps

    # Log errors
    error_log = []

    pressure_prev = None

    for t in tqdm(time_steps):
        full_network = copy.deepcopy(network)
        full_network.pressure_prev = (
            pressure_prev  # Nodal pressure values at previous time step
        )

        if pressure_prev is None:
            full_network.run_initialization = True
        else:
            full_network.run_initialization = False
            full_network.assign_pressure_values(pressure_prev)

        for i in full_network.nodes.keys():
            if i in full_network.reference_nodes:
                full_network.nodes[i].volumetric_flow = None
                full_network.nodes[i].energy_flow = None
            else:
                try:
                    if profile_type == "volumetric":
                        full_network.nodes[i].volumetric_flow = profiles[str(i)][t]
                        full_network.nodes[i].convert_volumetric_to_energy_flow()
                    elif profile_type == "energy":
                        full_network.nodes[i].energy_flow = profiles[str(i)][t]
                        full_network.nodes[i].convert_energy_to_volumetric_flow()
                    else:
                        raise ValueError(f"Unknown profile type {profile_type}!")
                    # full_network.nodes[i].demand_type = 'energy'
                except KeyError:
                    print(f"Node index {i} is not found!")
        # simplified_network = update_network_topology(full_network)
        simplified_network = full_network
        try:
            # network = run_snapshot(simplified_network)
            # for n in full_network.nodes.values():
            #     if n.volumetric_flow is not None and n.volumetric_flow < 0:
            #         print(n.volumetric_flow)
            full_network = copy.deepcopy(run_snapshot(full_network))
            pressure_prev = full_network.save_pressure_values()
        except RuntimeError:
            # error_log.append([simplified_network, profiles.iloc[t]])
            error_log.append([full_network, profiles.iloc[t]])

        results = save_time_series_results(full_network, results, results_to_save)
    # Save simulation results to file
    save_time_series_results_to_file(
        results, time_steps, output_format, output_filename
    )

    return results


def validate_results_to_save(results_to_save):
    """
    Validate the keys in results_to_save against VALID_RESULT_KEYS.
    :param results_to_save: List of result keys to save.
    :raises ValueError: If any key in results_to_save is invalid.
    """
    unrecognized_keys = [key for key in results_to_save if key not in VALID_RESULT_KEYS]
    if unrecognized_keys:
        raise ValueError(
            f"Unrecognized keys in results_to_save: {unrecognized_keys}. "
            f"Allowed keys are: {VALID_RESULT_KEYS}"
        )


def save_time_series_results(network, results, results_to_save):
    """
    Save simulation results dynamically based on the specified results_to_save keys.
    :param network: The network object after simulation for the current time step.
    :param results: Dictionary to store results for all time steps.
    :param results_to_save: List of result keys to save (e.g., 'nodal_pressure', 'pipeline_flowrate').
    :return: Updated results dictionary.
    """
    # Mapping of result keys to corresponding data extraction logic
    result_extraction_map = {
        "nodal_pressure": lambda: [node.pressure for node in network.nodes.values()],
        "nodal_gas_composition": lambda: [
            node.gas_mixture.composition for node in network.nodes.values()
        ],
        "pipeline_flowrate": lambda: [
            pipe.flow_rate for pipe in network.pipelines.values()
        ],
        "nodal_HHV_MJ_per_sm3": lambda: [
            node.gas_mixture.HHV_J_per_sm3 / 1e6 for node in network.nodes.values()
        ],
        "nodal_WI_MJ_per_sm3": lambda: [
            node.gas_mixture.WI_J_per_sm3 / 1e6 for node in network.nodes.values()
        ],
        "nodal_volume_flow_sm3_per_s": lambda: [
            node.volumetric_flow for node in network.nodes.values()
        ],
        "nodal_energy_flow_MW": lambda: [
            node.energy_flow for node in network.nodes.values()
        ],
    }

    # Validate keys (optional; useful if save_time_series_results is called independently)
    validate_results_to_save(results_to_save)

    # Dynamically append results based on keys in results_to_save
    for key in results_to_save:
        results[key].append(result_extraction_map[key]())

    return results


def save_time_series_results_to_file(
    results, time_steps, output_format="excel", output_filename="time_series_results"
):
    """
    Save simulation results to a file in the specified format.
    :param results: Dictionary containing results of simulation.
    :param time_steps: List of time step indices.
    :param output_format: Output format, options are 'excel', 'csv', 'hdf5', or 'json'.
    :param output_filename: Base filename for output files.
    """
    # Create DataFrames for each type of result
    nodal_pressure_df = pd.DataFrame(
        results["nodal_pressure"],
        index=time_steps,
        columns=[f"node_{i+1}" for i in range(len(results["nodal_pressure"][0]))],
    )
    pipeline_flowrate_df = pd.DataFrame(
        results["pipeline_flowrate"],
        index=time_steps,
        columns=[
            f"pipeline_{i+1}" for i in range(len(results["pipeline_flowrate"][0]))
        ],
    )
    nodal_gas_composition_df = pd.DataFrame(
        results["nodal_gas_composition"],
        index=time_steps,
        columns=[
            f"node_{i+1}" for i in range(len(results["nodal_gas_composition"][0]))
        ],
    )

    dataframes = {
        "Nodal Pressure": nodal_pressure_df,
        "Pipeline Flowrate": pipeline_flowrate_df,
        "Nodal Gas Composition": nodal_gas_composition_df,
    }

    # Handle output format
    if output_format.lower() == "excel":
        with pd.ExcelWriter(f"{output_filename}.xlsx") as writer:
            for sheet_name, df in dataframes.items():
                df.to_excel(writer, sheet_name=sheet_name)
        print(f"Results saved to {output_filename}.xlsx")

    elif output_format.lower() == "csv":
        os.makedirs(output_filename, exist_ok=True)
        for sheet_name, df in dataframes.items():
            df.to_csv(os.path.join(output_filename, f"{sheet_name}.csv"))
        print(f"Results saved to directory: {output_filename}")

    elif output_format.lower() == "hdf5":
        with pd.HDFStore(f"{output_filename}.h5") as store:
            for sheet_name, df in dataframes.items():
                store.put(sheet_name, df, format="table")
        print(f"Results saved to {output_filename}.h5")

    elif output_format.lower() == "json":
        os.makedirs(output_filename, exist_ok=True)
        for sheet_name, df in dataframes.items():
            df.to_json(
                os.path.join(output_filename, f"{sheet_name}.json"), orient="split"
            )
        print(f"Results saved to directory: {output_filename}")

    else:
        raise ValueError(
            f"Unsupported format: {output_format}. Supported formats are: 'excel', 'csv', 'hdf5', 'json'."
        )
