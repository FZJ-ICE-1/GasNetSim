#   #!/usr/bin/env python
#   -*- coding: utf-8 -*-
#   ******************************************************************************
#     Copyright (c) 2025.
#     Developed by Yifei Lu
#     Last change on 1/5/25, 11:15 PM
#     Last change by yifei
#    *****************************************************************************
import pandas as pd
from pathlib import Path
import logging
import copy
import os
from tqdm import tqdm
from typing import Dict, List

from ..components.network import Network
from ..components.utils.utils import plot_network_demand_distribution
from ..components.utils.cuda_support import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(level=logging.WARNING)

VALID_RESULT_KEYS = [
    "nodal_pressure",
    "nodal_gas_composition",
    "nodal_HHV_MJ_per_sm3",
    "nodal_WI_MJ_per_sm3",
    "nodal_volume_flow_sm3_per_s",
    "nodal_energy_flow_MW",
    "pipeline_flowrate",
    "pipeline_batch_locations",
    "pipeline_batch_hydrogen_fraction",
    "pipeline_outflow_hydrogen_fraction"
]


def read_profiles(file_path, sep=";"):
    """
    Read profiles from a CSV file and handle potential issues like index columns or non-integer column names.
    :param file_path: Path to the CSV file.
    :param sep: Separator used in the CSV file (default is ';').
    :return: DataFrame with profile data.
    """
    try:
        profiles = pd.read_csv(Path(file_path), sep=sep)

        # Handle potential index column (e.g., "Unnamed: 0")
        if "Unnamed: 0" in profiles.columns:
            profiles = profiles.set_index("Unnamed: 0")

        # Convert non-time column names to integers, if possible
        profiles.columns = [
            int(col) if col != "time" else col for col in profiles.columns
        ]

        logger.info(f"Successfully read profiles from {file_path}.")
        return profiles

    except FileNotFoundError:
        raise FileNotFoundError(f"The file at '{file_path}' was not found.")
    except pd.errors.EmptyDataError:
        raise ValueError(f"The file at '{file_path}' contains no data.")
    except pd.errors.ParserError as e:
        raise ValueError(f"Failed to parse the CSV file at '{file_path}': {str(e)}")
    except ValueError as e:
        raise ValueError(f"Column name conversion failed: {str(e)}")


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
    """
    Validate and preprocess the profiles DataFrame.
    :param profiles: DataFrame containing profile data.
    :return: Processed profiles DataFrame.
    """
    if "time" in profiles.columns:
        # Handle 'time' column if present
        if profiles["time"].dtype == int:
            # Convert integer-based time to datetime
            profiles["time"] = pd.to_datetime(
                profiles["time"], unit="s", origin=pd.Timestamp("1970-01-01")
            )
            logger.info("Converted integer 'time' column to datetime format.")
        elif pd.api.types.is_datetime64_any_dtype(profiles["time"]):
            logger.info("Time column is already in datetime format.")
        else:
            raise ValueError("The 'time' column must be either integer or datetime.")

        # Set 'time' as the index for time-based operations
        profiles = profiles.set_index("time")
        logger.info("Set 'time' column as index.")
    else:
        logger.info("'time' column is not present. Proceeding without time index.")

    return profiles


def run_snapshot(
        network, tol=0.01, max_iter=100, use_cuda=False, tracking_method="simple_mixing", time_step=3600
):
    # plot_network_demand_distribution(network)
    if use_cuda:
        is_cuda_available()
    network = network.simulation(
        tol=tol, max_iter=max_iter, use_cuda=use_cuda, tracking_method=tracking_method, time_step=time_step
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
        profiles,
        sep=";",
        profile_type="energy",
        tolerance=0.01,
        max_iter=100,
        time_step=3600,  # 1 hour
        tracking_method="simple_mixing",
        save_to_file=True,
        output_format="excel",
        output_filename="time_series_results",
        results_to_save=["nodal_pressure", "pipeline_flowrate", "nodal_gas_composition"],
        use_cuda=False,
):
    """
    Run time series simulation for the network and save results in specified format.
    """
    # Validate results_to_save before running the simulation
    validate_results_to_save(results_to_save)

    # Ensure profiles are not empty
    if profiles.empty:
        raise ValueError("Profiles data is empty. Please provide valid profile data.")

    # Initialize
    full_network = copy.deepcopy(network)
    results = dict([(k, []) for k in results_to_save])

    # # Read profiles
    # if file is not None:
    #     profiles = read_profiles(file, sep=sep)
    #     time_steps = profiles.index
    # else:
    #     time_steps = range(5)  # Test with 5 fictitious time steps
    time_steps = profiles.index

    # Log errors
    error_log = []

    pressure_prev = None

    full_network = copy.deepcopy(network)  # first time step

    for t in tqdm(time_steps):
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
            elif i in profiles.columns:  # Only apply profiles to nodes that have profile data
                if profile_type == "volumetric":
                    full_network.nodes[i].volumetric_flow = profiles.loc[t][i]
                    full_network.nodes[i].convert_volumetric_to_energy_flow()
                elif profile_type == "energy":
                    full_network.nodes[i].energy_flow = profiles.loc[t][i]
                    full_network.nodes[i].convert_energy_to_volumetric_flow()
                else:
                    raise ValueError(f"Unknown profile type {profile_type}!")
                # full_network.nodes[i].demand_type = 'energy'
            # If node is not a reference node and has no profile data, keep existing flow values
        # simplified_network = update_network_topology(full_network)
        simplified_network = full_network
        try:
            # network = run_snapshot(simplified_network)
            # for n in full_network.nodes.values():
            #     if n.volumetric_flow is not None and n.volumetric_flow < 0:
            #         print(n.volumetric_flow)
            full_network = copy.deepcopy(
                run_snapshot(
                    network=full_network,
                    tracking_method=tracking_method,
                    tol=tolerance,
                    max_iter=max_iter,
                    time_step=time_step,
                    use_cuda=use_cuda,
                )
            )
            pressure_prev = full_network.save_pressure_values()
        except (RuntimeError, TypeError) as e:
            # error_log.append([simplified_network, profiles.iloc[t]])
            print(e)
            error_log.append([e, full_network, profiles.loc[t]])

        results = save_time_series_results(full_network, results, results_to_save)
    # Save simulation results to file
    if save_to_file:
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
        "pipeline_batch_locations": lambda: [
            pipe.batch_location_history for pipe in network.pipelines.values()
        ],
        "pipeline_batch_hydrogen_fraction": lambda: [
            [composition[14] for composition in pipe.composition_history]
            for pipe in network.pipelines.values()
        ],
        "pipeline_outflow_hydrogen_fraction": lambda: [
            pipe.outflow_composition[14] for pipe in network.pipelines.values()
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
        results: Dict[str, List],
        time_steps: List[int],
        output_format: str = "excel",
        output_filename: str = "time_series_results",
):
    """
    Save network.results to a file in the specified format.
    :param results: Dictionary containing results of simulation (e.g., network.results).
    :param time_steps: List of time step indices.
    :param output_format: Output format, options are 'excel', 'csv', 'hdf5', or 'json'.
    :param output_filename: Base filename for output files.
    """
    # Dynamically create DataFrames
    dataframes = {}
    for key, data in results.items():
        if data:
            label = "node" if "nodal" in key else "pipeline"
            dataframes[key.replace("_", " ").title()] = pd.DataFrame(
                data,
                index=time_steps,
                columns=[f"{label}_{i + 1}" for i in range(len(data[0]))],
            )

    # Ensure output directory exists for formats that need it
    if output_format.lower() in {"csv", "json"}:
        os.makedirs(output_filename, exist_ok=True)

    # Define format-specific save logic
    save_methods = {
        "excel": lambda: save_to_excel(dataframes, output_filename),
        "csv": lambda: save_to_csv(dataframes, output_filename),
        "hdf5": lambda: save_to_hdf5(dataframes, output_filename),
        "json": lambda: save_to_json(dataframes, output_filename),
    }

    # Save results based on output_format
    try:
        save_method = save_methods.get(output_format.lower())
        if save_method:
            save_method()
        else:
            raise ValueError(
                f"Unsupported format: {output_format}. Supported formats are: 'excel', 'csv', 'hdf5', 'json'."
            )
    except Exception as e:
        raise IOError(f"Failed to save results: {str(e)}")


def save_to_excel(dataframes: Dict[str, pd.DataFrame], output_filename: str):
    with pd.ExcelWriter(f"{output_filename}.xlsx") as writer:
        for sheet_name, df in dataframes.items():
            df.to_excel(writer, sheet_name=sheet_name)
    print(f"Results saved to {output_filename}.xlsx")


def save_to_csv(dataframes: Dict[str, pd.DataFrame], output_filename: str):
    for sheet_name, df in dataframes.items():
        df.to_csv(os.path.join(output_filename, f"{sheet_name}.csv"))
    print(f"Results saved to directory: {output_filename}")


def save_to_hdf5(dataframes: Dict[str, pd.DataFrame], output_filename: str):
    with pd.HDFStore(f"{output_filename}.h5") as store:
        for sheet_name, df in dataframes.items():
            store.put(sheet_name, df, format="table")
    print(f"Results saved to {output_filename}.h5")


def save_to_json(dataframes: Dict[str, pd.DataFrame], output_filename: str):
    for sheet_name, df in dataframes.items():
        df.to_json(os.path.join(output_filename, f"{sheet_name}.json"), orient="split")
    print(f"Results saved to directory: {output_filename}")
