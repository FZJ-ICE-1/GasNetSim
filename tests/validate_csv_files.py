#!/usr/bin/env python
# coding: utf-8

# **********************************************************************************************************************
# This script ensures that the .csv file conforms to the established format guidelines.
# **********************************************************************************************************************

import csv
import logging
import os

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

nodes_expected_headers = [
    "node_index",
    "pressure_pa",
    "flow_sm3_per_s",
    "flow_MW",
    "altitude_m",
    "temperature_k",
    "gas_composition",
    "node_type",
    "flow_type",
    "longitude",
    " latitude",
    " remarks",
]

pipeline_expected_headers = [
    "pipeline_index",
    "inlet_index",
    "outlet_index",
    "diameter_m",
    "length_m",
    " is_bothDirection",
    "max_cap_M_m3_per_d",
    "max_pressure_bar",
    "remarks",
]

pipeline_mandatory_headers = [
    "pipeline_index",
    "inlet_index",
    "outlet_index",
    "diameter_m",
    "length_m",
]


def validate_pipelines_csv_format(file_path=None):
    """
    Validates the format of the pipelines CSV file.

    Args:
    - file_path (str): The path to the CSV file.

    Returns:
    - bool: True if the file conforms to the expected format, False otherwise.
    """

    # Check whether the file is present or not
    if not (os.path.isfile(file_path)):
        logger.info(f"The file pipelines CSV is either missing or not readable")
        return False

    with open(file_path, "r") as file:
        reader = csv.DictReader(file, delimiter=";")
        headers = reader.fieldnames
        print(headers)

        # Check if headers match the expected format
        if headers != pipeline_expected_headers:
            logger.info(f"The pipelines CSV file does not match the expected format.")
            return False

        # Check each row for missing values in required columns
        for row_num, row in enumerate(reader, start=1):
            missing_values = [
                key
                for key in pipeline_mandatory_headers
                if row.get(key, "").strip().lower() in ["", "nan", None]
            ]
            if missing_values:
                logger.info(
                    f"Missing value(s) in column(s) {', '.join(missing_values)} in row {row_num} "
                    f"in pipelines CSV."
                )
                return False

    logger.info(f"The pipelines CSV file follows the expected format.")
    return True


def validate_nodes_csv_format(file_path=None):
    """
    Validates the format of the nodes CSV file.

    Args:
    - file_path (str): The path to the CSV file.

    Returns:
    - bool: True if the file conforms to the expected format, False otherwise.
    """

    # Check if the file exists
    if not os.path.isfile(file_path):
        logger.info(f"The nodes file is either missing or not readable")
        return False

    with open(file_path, "r") as file:
        reader = csv.DictReader(file, delimiter=";")
        headers = reader.fieldnames

        # Check if headers match the expected format
        if headers != nodes_expected_headers:
            logger.info(f"The nodes CSV file does not match the expected format.")
            return False

        reference_found = False
        for row_num, row in enumerate(reader, start=1):

            # Check each row to see if at-least one 'reference' value is present or not
            if row["node_type"].lower() == "reference":
                reference_found = True

            # Check each row to see if any 'node_index' value is missing
            if not row["node_index"]:
                logger.info(f"Missing 'node_index' in row {row_num}.")
                return False

            # Check if either 'pressure_pa', 'flow_sm3_per_s' or 'flow_MW' value are present in the row
            if row["node_type"].lower() == "reference":
                if not row["pressure_pa"]:
                    logger.info(
                        f"The 'pressure_pa' value is missing for the 'reference node' located in row {row_num} "
                        f"of the nodes CSV file."
                    )
                    return False
            elif not row["flow_sm3_per_s"]:
                if not row["flow_MW"]:
                    logger.info(
                        f"Either the 'flow_sm3_per_s' or 'flow_MW' value is missing "
                        f"from row {row_num} of the nodes CSV file. This data is required."
                    )
                    return False
            if row["pressure_pa"] and (row["flow_sm3_per_s"] or row["flow_MW"]):
                logger.info(
                    f"The 'pressure_pa' field cannot be assigned a value when either the "
                    f"'flow_sm3_per_s' or 'flow_MW' field is provided. This constraint was "
                    f"violated in row {row_num}."
                )
                return False
            if row["flow_sm3_per_s"] and row["flow_MW"]:
                logger.info(
                    f"The 'flow_sm3_per_s' field cannot be assigned a value together with the "
                    f"'flow_MW' field. This constraint was violated in row {row_num}."
                )
                return False

        if not reference_found:
            logger.info(
                f"No 'reference' value found in the 'node_type' column found in nodes CSV."
            )
            return False

    logger.info(f"The nodes CSV file follows the expected format.")
    return True


# Check if the script is being run directly
if __name__ == "__main__":
    # Input the 'File_path' with the path of the pipeline CSV file
    File_path = input(
        "Enter the path of the .csv file.\nFor example "
        "'../examples/Irish13/Irish13_pipelines.csv':"
    )

    validate_pipelines_csv_format(file_path=File_path)
    validate_nodes_csv_format(file_path=File_path)
