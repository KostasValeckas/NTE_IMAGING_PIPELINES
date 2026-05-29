from datetime import datetime
import numpy as np
import os
from astropy.io import fits
from enum import Enum
from logging import Logger
from datatypes import Processed_frame
import matplotlib.pyplot as plt


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


def write_frame(
    instrument,
    hdul,
    master_frame,
    write_name,
    output_path,
    logger: Logger,
    bad_pixel_mask=None,
    comment=None,
    header_updates=None,
):
    """
    header_updates should be a dict of {keyword: value} pairs to update in the header of the data HDU, if provided. If not provided, no header updates will be made.
    """

    # print type of maskter bias and bad pixel mask for debugging
    print(f"Type of master_frame: {type(master_frame)}, shape: {master_frame.shape}")
    print(f"Type of bad_pixel_mask: {type(bad_pixel_mask)}")

    data_hdu = hdul[instrument.data_hdu_extension]

    data_hdu.data = master_frame.copy()

    # append the bad pixel mask as a new HDU to the HDUList
    orig_header = hdul[instrument.data_hdu_extension].header

    # TODO: the following code assumes that bpm hdul is always data + 1
    # and masked_array hdul is data + 2 - make more flexible?

    if bad_pixel_mask is not None:
        bad_pixel_hdu = fits.ImageHDU(
            data=bad_pixel_mask.astype(np.uint8), name="BAD_PIXEL_MASK"
        )

        # ensure the extension name is correct
        bad_pixel_hdu.header["EXTNAME"] = "BAD_PIXEL_MASK"

        bad_pixel_hdu.header.add_comment("Bad pixel mask for the frame")
        # if there exists a bpm - override
        if len(hdul) >= instrument.data_hdu_extension + 2:
            hdul[instrument.data_hdu_extension + 1] = bad_pixel_hdu
        # else - create a bpm entry
        else:
            hdul.append(bad_pixel_hdu)

        # mask the data if bpm is provided and append to the hdul

        masked_data_hdul = data_hdu.copy()
        masked_data_hdul.data = master_frame.copy()
        masked_data_hdul.data[np.array(bad_pixel_hdu.data, dtype=bool)] = np.nan
        masked_data_hdul.header["EXTNAME"] = "MASKED_FRAME"

        hdul.append(masked_data_hdul)

    elif len(hdul) >= instrument.data_hdu_extension + 3:

        bad_pixel_hdu = hdul[instrument.data_hdu_extension + 1]

        masked_data_hdul = hdul[instrument.data_hdu_extension + 2]
        masked_data_hdul.data = master_frame.copy()
        masked_data_hdul.data[np.array(bad_pixel_hdu.data, dtype=bool)] = np.nan
        masked_data_hdul.header["EXTNAME"] = "MASKED_FRAME"

    # update header to record creation
    try:
        data_hdu.header.add_history(f"Created: {datetime.utcnow().isoformat()} UTC")

        if comment is not None:
            data_hdu.header.add_comment(comment)

        if header_updates is not None:
            for key, value in header_updates.items():
                data_hdu.header[key] = value

    except Exception as e:
        # push through header update errors
        logger.warning(
            f"Failed to update header for {write_name} due to {e}, proceeding without header updates."
        )
        pass

    output_path = os.path.join(output_path, write_name)

    try:
        hdul.writeto(output_path, overwrite=True)
        logger.info(f"Successfully wrote frame to {output_path}")
    except Exception as e:
        logger.error(f"Error writing frame to {output_path}: {e}")

    # close HDULists but no biggie if hiccup
    try:
        hdul.close()
    except Exception as e:
        logger.warning(f"Error closing HDUList after writing {output_path}: {e}")
        pass


def read_frame(output_path, name, instrument, logger: Logger):

    file_path = os.path.join(output_path, name)

    try:
        hdul = open_fits_file(file_path, logger).copy()

    except Exception as e:
        logger.error(f"Error reading frame from {file_path}: {e}")
        return None

    if hdul is None:
        logger.error(f"Failed to read frame from {file_path}, HDUList is None.")
        return None

    frame_data = hdul[instrument.data_hdu_extension].data
    bpm = hdul["BAD_PIXEL_MASK"].data if "BAD_PIXEL_MASK" in hdul else None

    return Processed_frame(hdul, frame_data, bpm)
