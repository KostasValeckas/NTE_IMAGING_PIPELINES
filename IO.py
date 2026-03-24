import numpy as np
import os
from astropy.io import fits
from enum import Enum

from instruments import Instrument
from datetime import datetime
from datatypes import ImageType
from typing import Optional


def sort_data(instrument: Instrument, raw_data_path: str, output_path: str = None):

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
        hdul = open_fits_file(filepath)
        if hdul is None:
            Instrument.logger.warning(
                f"Could not open {filename} to read headers, skipping"
            )
            continue
        try:
            image_type = Instrument.match_image_type(hdul)
            Instrument.logger.info(
                f"File: {filename}, matched image type: {image_type}"
            )

        except Exception as e:

            Instrument.logger.warning(f"match_image_type failed for {filename}: {e}")
            continue

        if image_type == ImageType.BIAS:
            bias_files.append(filepath)
        elif image_type == ImageType.DARK:
            dark_files.append(filepath)
        elif image_type == ImageType.FLAT:
            flat_files.append(filepath)
        elif image_type == ImageType.SCIENCE:
            science_files.append(filepath)
        else:
            Instrument.logger.warning(f"Unknown image type for {filename}")

    Instrument.logger.info(
        f"Sorted files: {len(bias_files)} bias, {len(dark_files)} dark, "
        f"{len(flat_files)} flat, {len(science_files)} science"
    )

    # write the lists to the disc

    date = datetime.now().strftime("%Y-%m-%d")

    # if no output path is provided, use the raw data path,
    if output_path is None or output_path == "":
        output_path = raw_data_path

    bias_filename = os.path.join(output_path, f"bias_files_{date}.txt")
    dark_filename = os.path.join(output_path, f"dark_files_{date}.txt")
    flat_filename = os.path.join(output_path, f"flat_files_{date}.txt")
    science_filename = os.path.join(output_path, f"science_files_{date}.txt")

    write_list = [bias_files, dark_files, flat_files, science_files]

    for file_list, filename in zip(
        write_list, [bias_filename, dark_filename, flat_filename, science_filename]
    ):
        with open(filename, "w") as f:
            for item in file_list:
                f.write(f"{item}\n")

    return bias_files, dark_files, flat_files, science_files


def open_fits_file(filepath: str):
    # a robust wrapper around fits.open that handles exceptions and logs errors
    try:
        hdul = fits.open(filepath)
        return hdul
    except Exception as e:
        Instrument.logger.error(f"Error opening FITS file {filepath}: {e}")
        return None



def get_header_value(hdul, keyword_tuple) -> Optional[str]:
    """Helper method to extract header value based on provided keyword tuple"""
    if keyword_tuple is None or len(keyword_tuple) != 2:
        return None
    
    keys, hdu_index = keyword_tuple
    if not isinstance(keys, list):
        keys = [keys]
    
    for key in keys:
        try:
            value = hdul[hdu_index].header[key]
            if value is not None:
                return value
        except KeyError:
            continue
    
    return None

def get_header_values(hdul, keyword_tuple) -> Optional[dict[str, Optional[str]]]:
    """Helper method to extract multiple header values based on provided keyword tuple"""
    if keyword_tuple is None or len(keyword_tuple) != 2:
        return None
    
    keys, hdu_index = keyword_tuple
    if not isinstance(keys, list):
        keys = [keys]
    
    values = []
    for key in keys:
        try:
            value = hdul[hdu_index].header[key]
            values.append(value)
        except KeyError:
            values.append(None)
    return values