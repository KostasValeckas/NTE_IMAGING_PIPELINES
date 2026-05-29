import numpy as np
import os
from astropy.io import fits
from enum import Enum
from logging import Logger
import json
from instruments import Instrument
from datetime import datetime
from datatypes import ImageType
from IO import open_fits_file, get_header_value, get_header_values

"""
Module for automatic sorting of raw data.
"""


def sort_data(
    instrument: Instrument, logger: Logger, raw_data_path: str, output_path: str = None
):
    
    """
    Method for sorting raw data files into bias, dark, flat and science 
    categories based on criteria defined in the `Instrument` class 
    implementation of the instrument being processed.

    Writes lists to disc with the filenames of different categories, 
    and returns them as lists.

    Parameters
    ----------
    instrument : Instrument
        The instrument for which the data reduction is being run. The sorting
        relies on the `match_image_type` method of the instrument to classify
        the files.

    logger : Logger
        Logger object for logging messages during the sorting process.

    raw_data_path : str
        Path to the directory containing the raw data files to be sorted.

    output_path : str
        Path to the directory where the sorted file lists will be saved.

    Returns
    -------
    bias_files : list of str
        List of filenames classified as bias frames.

    dark_files : list of str
        List of filenames classified as dark frames.    

    flat_files : list of str
        List of filenames classified as flat frames.

    science_files : list of str
        List of filenames classified as science frames.
    """

    all_files = os.listdir(raw_data_path)

    bias_files = []
    dark_files = []
    flat_files = []
    science_files = []

    for filename in all_files:
        # Skip non-FITS files
        if not filename.lower().endswith((".fits", ".fit", ".fts")):
            continue
        filepath = os.path.join(raw_data_path, filename)

        # Open file and get HDUList (match_image_type now expects the whole HDU/HDUList)
        hdul = open_fits_file(filepath, logger)
        if hdul is None:
            logger.warning(f"Could not open {filename} to read headers, skipping")
            continue
        try:
            image_type = instrument.match_image_type(hdul)
            logger.info(f"File: {filename}, matched image type: {image_type}")

        except Exception as e:

            logger.warning(f"match_image_type failed for {filename}: {e}")
            continue

        filepath_stripped = os.path.relpath(filepath, raw_data_path)

        if image_type == ImageType.BIAS:
            bias_files.append(filepath_stripped)
        elif image_type == ImageType.DARK:
            dark_files.append(filepath_stripped)
        elif image_type == ImageType.FLAT:
            flat_files.append(filepath_stripped)
        elif image_type == ImageType.SCIENCE:
            science_files.append(filepath_stripped)
        else:
            logger.warning(f"Unknown image type for {filename}")

    # write the lists to the disc

    date = datetime.now().strftime("%Y-%m-%d")

    # if no output path is provided, use the raw data path,
    if output_path is None or output_path == "":
        output_path = raw_data_path

    bias_filename = os.path.join(output_path, f"bias_files_{date}.dat")
    dark_filename = os.path.join(output_path, f"dark_files_{date}.dat")
    flat_filename = os.path.join(output_path, f"flat_files_{date}.dat")
    science_filename = os.path.join(output_path, f"science_files_{date}.dat")

    write_list = [bias_files, dark_files, flat_files, science_files]

    for file_list, filename in zip(
        write_list, [bias_filename, dark_filename, flat_filename, science_filename]
    ):
        # If file exists, warn and remove it before creating a new one
        if os.path.exists(filename):
            logger.warning(f"Output file {filename} already exists; replacing it.")
            try:
                os.remove(filename)
            except Exception as e:
                logger.error(f"Could not remove existing file {filename}: {e}")

        try:
            with open(filename, "w") as f:
                for item in file_list:
                    f.write(f"{item}\n")
        except Exception as e:
            logger.error(f"Failed to write to {filename}: {e}")

    logger.info(
        f"Sorted files: {len(bias_files)} bias, {len(dark_files)} dark, "
        f"{len(flat_files)} flat, {len(science_files)} science"
    )

    return bias_files, dark_files, flat_files, science_files


def create_setup_table(
    instrument: Instrument,
    logger: Logger,
    input_dir: str,
    output_dir: str,
    science_file_list,
):
    
    """
    Sorts all science observations into setups based on
    -  the window used
    -  the binning in x and y
    -  the filter(s) used

    Parameters
    ----------
    instrument : Instrument
        The instrument for which the data reduction is being run.

    logger : Logger
        Logger instance for logging messages.

    input_dir : str
        Directory containing the input science files.

    output_dir : str
        Directory where the output setup table will be written.

    science_file_list : list of str
        List of science file names to process.

    Writes the setup to disc as a .json file.

    Returns
    -------
    setup_table : dict
        Dictionary containing the setup table information. The format is:
        {
            setup_idx: {
                "window": window,
                "bin_x": bin_x,
                "bin_y": bin_y,
                "filter": filter_names,
                "files": [list of science files belonging to this setup]
            },
            ...
        }
    """

    setup_table = {}
    setup_counter = 0
    key_map = {}

    for file in science_file_list:
        logger.info(f"Processing science file for setup table: {file}")

        path = os.path.join(input_dir, file)

        hdul = open_fits_file(path, logger)

        window = get_header_value(hdul, instrument.detector.window_keyword, logger)

        bin_x = get_header_value(hdul, instrument.detector.bin_x_keyword, logger)

        bin_y = get_header_value(hdul, instrument.detector.bin_y_keyword, logger)

        filter_names = get_header_values(hdul, instrument.filter_keyword, logger)

        key = (window, bin_x, bin_y, *filter_names)

        # If we've already seen this configuration, append the file to its list
        if key in key_map:
            idx = key_map[key]
            setup_table[idx]["files"].append(file)
        else:
            idx = setup_counter
            key_map[key] = idx
            setup_table[idx] = {
                "window": window,
                "bin_x": bin_x,
                "bin_y": bin_y,
                "filter": filter_names,
                "files": [file],
            }
            setup_counter += 1

    logger.info(f"Identified {len(setup_table)} unique setups")

    # write the setup table to a json file

    setup_table_filename = os.path.join(output_dir, "setup_table.json")

    try:
        with open(setup_table_filename, "w") as f:
            json.dump(setup_table, f, indent=4)
            logger.info(f"Setup table written to {setup_table_filename}")
    except Exception as e:
        logger.error(f"Failed to write setup table to {setup_table_filename}: {e}")

    return setup_table


# TODO see if it makes sense to refactor table crreation into one generic


def create_bias_table(
    instrument: Instrument,
    logger: Logger,
    input_dir: str,
    output_dir: str,
    setup_table,
    bias_file_list,
):

    """
    Creates a bias table that maps each unique combination of:
    -  window
    -  binning in x and y

    Also creates a map from science setups to bias table entries, as bias 
    is independent of filter so setups do not map 1-to-1.

    Both returns the setup dictionaries and writes them to disc as .json files.
    
    Parameters
    ----------
    instrument : Instrument
        The instrument for which the data reduction is being run.

    logger : Logger
        Logger instance.

    input_dir : str
        Directory containing the input bias files.

    output_dir : str
        Directory where the output bias table and science-to-bias map will be written.

    Returns
    -------
    bias_table : dict
        Dictionary containing the bias table information. The format is:
        {
            bias_idx: {
                "window": window,
                "bin_x": bin_x,
                "bin_y": bin_y,
                "files": [list of bias files belonging to this bias configuration]
            },
            ...
        }

    science_to_bias_map : dict
        Dictionary mapping science setup indices to bias table indices. The format is:
        {
            science_setup_idx: bias_idx,
            ...
        }
    """

    bias_table = {}
    bias_key_map = {}
    science_to_bias_map = {}
    bias_table_counter = 0

    # first - create the tables without sorting bias files. Slower, but
    # simpler to implement and less error-prone.
    for setup_idx, setup in setup_table.items():

        bias_key = (setup["window"], setup["bin_x"], setup["bin_y"])

        if bias_key in bias_key_map:
            bias_idx = bias_key_map[bias_key]
            science_to_bias_map[setup_idx] = bias_idx
        else:
            bias_key_map[bias_key] = bias_table_counter
            bias_table[bias_table_counter] = {
                "window": setup["window"],
                "bin_x": setup["bin_x"],
                "bin_y": setup["bin_y"],
                "files": [],
            }
            science_to_bias_map[setup_idx] = bias_key_map[bias_key]
            bias_table_counter += 1

    # now loop through bias files and assign them to the correct bias table entry
    for bias_file in bias_file_list:

        path = os.path.join(input_dir, bias_file)

        hdul = open_fits_file(path, logger)

        window = get_header_value(hdul, instrument.detector.window_keyword, logger)

        bin_x = get_header_value(hdul, instrument.detector.bin_x_keyword, logger)

        bin_y = get_header_value(hdul, instrument.detector.bin_y_keyword, logger)

        bias_key = (window, bin_x, bin_y)

        if bias_key in bias_key_map:
            bias_idx = bias_key_map[bias_key]
            bias_table[bias_idx]["files"].append(bias_file)
        else:
            logger.warning(
                f"Bias file {bias_file} has no matching setup key {bias_key}"
            )

    # write the bias table and the map to science to disc

    bias_table_filename = os.path.join(output_dir, "bias_table.json")
    science_to_bias_filename = os.path.join(output_dir, "science_to_bias_map.json")

    try:
        with open(bias_table_filename, "w") as f:
            json.dump(bias_table, f, indent=4)
            logger.info(f"Bias table written to {bias_table_filename}")
    except Exception as e:
        logger.error(f"Failed to write bias table to {bias_table_filename}: {e}")

    try:
        with open(science_to_bias_filename, "w") as f:
            json.dump(science_to_bias_map, f, indent=4)
            logger.info(f"Science to bias map written to {science_to_bias_filename}")
    except Exception as e:
        logger.error(
            f"Failed to write science to bias map to {science_to_bias_filename}: {e}"
        )

    return bias_table, science_to_bias_map


def create_flat_table(
    instrument: Instrument,
    logger: Logger,
    input_dir: str,
    output_dir: str,
    setup_table,
    flat_file_list,
):
    
    """
    Creates a flat table that maps each unique combination of:
    -  window
    -  binning in x and y
    -  filter(s) used

    Returns the flat table as a dictionary and writes it to disc as a .json file.

    Parameters
    ----------
    instrument : Instrument
        The instrument for which the data reduction is being run.

    logger : Logger
        Logger instance for logging messages.

    input_dir : str
        Directory containing the input files.

    output_dir : str
        Directory where the output files will be written.

    setup_table : dict
        Dictionary mapping setup indices to their parameters.

    flat_file_list : list
        List of flat field files to process.

    Returns
    -------
    flat_table : dict
        Dictionary mapping setup indices to their flat field parameters. The format is:
        {
            setup_idx: {
                "window": window,
                "bin_x": bin_x,
                "bin_y": bin_y,
                "filter": filter_names,
                "files": [list of flat files belonging to this setup]
            },
            ...
        }
    """

    flat_table = {}

    for setup_idx, setup in setup_table.items():

        # flats setup match science directly, so we just copy over
        flat_table[setup_idx] = {
            "window": setup["window"],
            "bin_x": setup["bin_x"],
            "bin_y": setup["bin_y"],
            "filter": setup["filter"],
            "files": [],
        }

    for flat_file in flat_file_list:

        path = os.path.join(input_dir, flat_file)

        hdul = open_fits_file(path, logger)

        window = get_header_value(hdul, instrument.detector.window_keyword, logger)

        bin_x = get_header_value(hdul, instrument.detector.bin_x_keyword, logger)

        bin_y = get_header_value(hdul, instrument.detector.bin_y_keyword, logger)

        filter_names = get_header_values(hdul, instrument.filter_keyword, logger)

        flat_key = (window, bin_x, bin_y, filter_names)

        # find the matching setup index
        matched_setup_idx = None
        for setup_idx, setup in setup_table.items():
            setup_key = (
                setup["window"],
                setup["bin_x"],
                setup["bin_y"],
                setup["filter"],
            )
            if flat_key == setup_key:
                matched_setup_idx = setup_idx
                break

        if matched_setup_idx is not None:
            flat_table[matched_setup_idx]["files"].append(flat_file)
        else:
            logger.warning(
                f"Flat file {flat_file} has no matching setup key {flat_key}"
            )

    # write the flat table to disc
    flat_table_filename = os.path.join(output_dir, "flat_table.json")

    try:
        with open(flat_table_filename, "w") as f:
            json.dump(flat_table, f, indent=4)
            logger.info(f"Flat table written to {flat_table_filename}")
    except Exception as e:
        logger.error(f"Failed to write flat table to {flat_table_filename}: {e}")

    return flat_table
