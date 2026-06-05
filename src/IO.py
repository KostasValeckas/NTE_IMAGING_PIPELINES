from datetime import datetime
import numpy as np
import os
from astropy.io import fits
from logging import Logger
from datatypes import Processed_frame


"""
Module for general IO (mostly FITS file handling).
"""


def open_fits_file(filepath: str, logger: Logger):
    """
    A robust wrapper around fits.open that handles exceptions and logs errors.

    Parameters
    ----------
    filepath : str
        The path to the FITS file to be opened.
    logger : Loggers
        The logger instance to use for logging errors.

    Returns
    -------
    hdul : astropy.io.fits.HDUList or None
        The opened HDUList if successful, or None if an error occurred.
    """
    try:
        hdul = fits.open(filepath)
        return hdul
    except Exception as e:
        logger.error(f"Error opening FITS file {filepath}: {e}")
        return None


def get_header_value(hdul, keyword_tuple, logger: Logger):
    """
    Helper method to extract header value based on provided keyword tuple
    for singular values.
    (see `Instrument` class for details on keyword tuple structure).

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        The HDUList from which to extract the header value.

    keyword_tuple : tuple
        A tuple of the form (keyword, hdu_index) where:
        - keyword: The header keyword to extract.
        - hdu_index: The index of the HDU from which to extract the header values.

    logger : Logger
        The logger instance to use for logging warnings or errors.

    Returns
    -------
    value : str or None
        The extracted header value if successful,
        "CONSTANT" if the keyword indicates a constant value,
        or None if the keyword tuple is invalid or the header value is not found.

    """
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
    """
    Helper method to extract header value based on provided keyword tuple
    for a list of values.
    (see `Instrument` class for details on keyword tuple structure).

    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        The HDUList from which to extract the header value.

    keyword_tuple : tuple
        A tuple of the form (keyword, hdu_index) where:
        - keyword: The header keyword to extract.
        - hdu_index: The index of the HDU from which to extract the header values.

    logger : Logger
        The logger instance to use for logging warnings or errors.

    Returns
    -------
    value : str or None
        The extracted header value if successful,
        "CONSTANT" if the keyword indicates a constant value,
        or None if the keyword tuple is invalid or the header value is not found.

    """

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
    sigma_array = None,
    bad_pixel_mask=None,
    comment=None,
    header_updates=None,
):
    """
    A method for writing frames to disc that also encapsulates the functionality
    of extending the existing FITS header with additional HDUs (e.g. bad pixel mask)
    and updating the header with relevant information.

    Parameters
    ----------
    instrument : Instrument
        The `Instrument` object.

    hdul : astropy.io.fits.HDUList
        The HDUList to which the master frame and additional HDUs will be written.

    master_frame : numpy.ndarray
        The master frame data to be written to the FITS file.

    write_name : str
        The name of the output FITS file (without path).

    output_path : str
        The directory path where the output FITS file will be saved.

    logger : Logger
        The logger instance to use for logging information and errors.

    bad_pixel_mask : numpy.ndarray, optional
        A 2D array representing the bad pixel mask, where bad pixels are marked as True
        and good pixels as False. If provided, this mask will be written as a new HDU
        and the master frame will be masked accordingly. If not provided, no bad pixel mask
        will be written and the master frame will not be masked.
    comment : str, optional
        A comment to be added to the header of the data HDU, if provided. If
        not provided, no comment will be added.

    header_updates : dict, optional
        should be a dict of {keyword: value} pairs to update in the
        header of the data HDU, if provided. If not provided, no header updates
        will be made.

        The format is {keyword: value} where keyword is the header keyword
        to be updated and value is the new value to be set for that keyword and a comment
        can be optionally provided as a tuple (value, comment) where comment is a string
        to be added as a comment for that header keyword.
    """

    data_hdu = hdul[instrument.data_hdu_extension]

    data_hdu.data = master_frame.copy()

    # TODO: a lot of repeated code, could be refactored, especially for 
    # existing entry checking

    # check bools for handling existing HDUs - if they exist, override, if not - append
    sigma_exists = False
    bpm_exists = False
    new_bpm = False


    try:
        _ = hdul["BAD_PIXEL_MASK"]
        bpm_exists = True
    except IndexError:
        logger.info("No existing BAD_PIXEL_MASK HDU found, will create new one.")
    except Exception as e:
        logger.warning(f"Error checking for existing BAD_PIXEL_MASK HDU: {e}, will create new one.")



    try:
        _ = hdul["ERROR"]
        sigma_exists = True
    except IndexError:
        logger.info("No existing ERROR HDU found, will create new one.")
    except Exception as e:
        logger.warning(f"Error checking for existing ERROR HDU: {e}, will create new one.")


    if sigma_array is not None:
        sigma_hdu = fits.ImageHDU(
            data=sigma_array.astype(np.float32), name="ERROR"
        )

        if sigma_exists:
            hdul["ERROR"] = sigma_hdu
        else:
            hdul.append(sigma_hdu)


    if bad_pixel_mask is not None:
        bad_pixel_hdu = fits.ImageHDU(
            data=bad_pixel_mask.astype(np.uint8), name="BAD_PIXEL_MASK"
        )

        # ensure the extension name is correct
        bad_pixel_hdu.header["EXTNAME"] = "BAD_PIXEL_MASK"

        bad_pixel_hdu.header.add_comment("Bad pixel mask for the frame")


        if bpm_exists:
            hdul["BAD_PIXEL_MASK"] = bad_pixel_hdu

        else:
            hdul.append(bad_pixel_hdu)
            new_bpm = True

        bpm_exists = True

    if bpm_exists:

        # mask the data and append that also

        bad_pixel_hdu = hdul["BAD_PIXEL_MASK"]

        masked_data_hdul = data_hdu.copy()
        masked_data_hdul.data = master_frame.copy()
        masked_data_hdul.data[np.array(bad_pixel_hdu.data, dtype=bool)] = np.nan
        masked_data_hdul.header["EXTNAME"] = "MASKED_FRAME"

        # if bpm exists, assume masked data array exists and override, if not - append
        if not new_bpm:
            hdul["MASKED_FRAME"] = masked_data_hdul
        else:
            hdul.append(masked_data_hdul)

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
    uncertainty = hdul["ERROR"].data if "ERROR" in hdul else None

    return Processed_frame(hdul, frame_data, bpm, uncertainty)
