import numpy as np
import os
from astropy.io import fits
from enum import Enum
from logging import Logger



def open_fits_file(filepath: str, logger: Logger):
    # a robust wrapper around fits.open that handles exceptions and logs errors
    try:
        hdul = fits.open(filepath)
        return hdul
    except Exception as e:
        logger.error(f"Error opening FITS file {filepath}: {e}")
        return None


def get_header_value(hdul, keyword_tuple, logger: Logger):
    """Helper method to extract header value based on provided keyword tuple"""
    if keyword_tuple is None or len(keyword_tuple) != 2:
        return None

    key, hdu_index = keyword_tuple

    # special case were header does not exist 
    if (key == "CONSTANT") or (key is None):
        return "CONSTANT"

    value = hdul[hdu_index].header[key]

    if value is not None:
        return value

    logger.info(
        f"Header value for key '{key}' in HDU index {hdu_index} is None or not found."
    )

    return None


def get_header_values(hdul, keyword_tuple, logger: Logger):
    """Helper method to extract multiple header values based on provided keyword tuple"""
    if keyword_tuple is None or len(keyword_tuple) != 2:
        return None

    keys, hdu_index = keyword_tuple

    # special case were header does not exist
    if (keys == "CONSTANT") or (keys is None):
        return ["CONSTANT"]

    if not isinstance(keys, list):
        keys = [keys]

    values = []
    for key in keys:
        value = hdul[hdu_index].header[key]
        if value is None:
            logger.warning(
                f"Header value for key '{key}' in HDU index {hdu_index} is None."
            )
        values.append(value)

    return values
