from instruments import Instrument
import os
from logger import init_logger
from astropy.io import fits
import numpy as np

from instruments import ImageType


class ReductionPipeline:
    def __init__(self, instrument: Instrument, raw_data_path):

        self.logger = init_logger(raw_data_path)

        self.instrument = instrument
        self.raw_data_path = raw_data_path

        # to be initialized later
        self.bias_files = None
        self.dark_files = None
        self.flat_files = None
        self.science_files = None

        self.setup_table = {}

    def sort_data(self):

        all_files = os.listdir(self.raw_data_path)

        bias_files = []
        dark_files = []
        flat_files = []
        science_files = []

        for filename in all_files:
            # Skip non-FITS files
            if not filename.lower().endswith((".fits", ".fit", ".fts")):
                continue

            filepath = os.path.join(self.raw_data_path, filename)

            header_extension_imtype = self.instrument.imagetype_keyword[1]
            header_extension_obsmode = self.instrument.obsmode_keyword[1]

            # Get header to determine image type
            header_imtype = self.get_fits_header(filepath, header_extension_imtype)
            header_obsmode = self.get_fits_header(filepath, header_extension_obsmode)

            if header_imtype is None or header_obsmode  is None:
                self.logger.warning(f"Could not read header from {filename}, skipping")
                continue

            imagetype_keyword = self.instrument.imagetype_keyword[0]
            obsmode_keyword = self.instrument.obsmode_keyword[0]

            image_type = self.instrument.match_image_type(
                header_imtype.get(imagetype_keyword, ""), header_obsmode.get(obsmode_keyword, "")
            )

            if image_type == ImageType.BIAS:
                bias_files.append(filepath)
            elif image_type == ImageType.DARK:
                dark_files.append(filepath)
            elif image_type == ImageType.FLAT:
                flat_files.append(filepath)
            elif image_type == ImageType.SCIENCE:
                science_files.append(filepath)
            else:
                self.logger.warning(f"Unknown image type for {filename}")

        self.bias_files = bias_files.copy()
        self.dark_files = dark_files.copy()
        self.flat_files = flat_files.copy()
        self.science_files = science_files.copy()

        self.logger.info(
            f"Sorted files: {len(bias_files)} bias, {len(dark_files)} dark, "
            f"{len(flat_files)} flat, {len(science_files)} science"
        )

    def create_setup_table(self):
        
        pass

        setup_table = {}

        setup_counter = 0

        for file in self.science_files:

            hdul = self.open_fits_file(file)

            if hdul is None:
                self.logger.warning(f"Could not open {file}, skipping")
                continue

            window_header_extension = self.instrument.detector.window_keyword[1]
            bin_x_header_extension = self.instrument.detector.bin_x_keyword[1]
            bin_y_header_extension = self.instrument.detector.bin_y_keyword[1]
            filter_header_extension = self.instrument.filter_keyword[1]

            window_keyword = self.instrument.detector.window_keyword[0]
            bin_x_keyword = self.instrument.detector.bin_x_keyword[0]
            bin_y_keyword = self.instrument.detector.bin_y_keyword[0]
            filter_keyword = self.instrument.filter_keyword[0]

            window = hdul[window_header_extension].header.get(window_keyword, "UNKNOWN")
            bin_x = hdul[bin_x_header_extension].header.get(bin_x_keyword, "UNKNOWN")
            bin_y = hdul[bin_y_header_extension].header.get(bin_y_keyword, "UNKNOWN")
            header = hdul[filter_header_extension].header
            filter_names = []
            if isinstance(filter_keyword, (list, tuple)):
                for key in filter_keyword:
                    val = header.get(key)
                    filter_names.append(str(val).strip() if val not in (None, "") else "UNKNOWN")
            else:
                val = header.get(filter_keyword, "UNKNOWN")
                filter_names.append(str(val).strip() if val not in (None, "") else "UNKNOWN")
            filter_name = filter_names

            print(f"File: {file}, Window: {window}, Bin X: {bin_x}, Bin Y: {bin_y}, Filter: {filter_name}")

            test_dict = {
                "window": window,
                "bin_x": bin_x,
                "bin_y": bin_y,
                "filter": filter_name,
            }

            if test_dict not in setup_table.values():
                setup_table[setup_counter] = test_dict
                setup_counter += 1

        self.logger.info(f"Created setup table with {len(setup_table)} unique setups")
        self.logger.info(f"Setup table contents: {setup_table}")   

            

        self.setup_table = setup_table


    def open_fits_file(self, filepath):
        """
        Open a FITS file and return the HDU list

        Args:
            filepath (str): Path to the FITS file

        Returns:
            astropy.io.fits.HDUList: The opened FITS file
        """
        try:
            with fits.open(filepath) as hdul:
                # Make a copy to return since we're using context manager
                hdul_copy = fits.HDUList([hdu.copy() for hdu in hdul])
                self.logger.info(f"Successfully opened FITS file: {filepath}")
                return hdul_copy
        except Exception as e:
            self.logger.error(f"Error opening FITS file {filepath}: {e}")
            return None

    def get_fits_header(self, filepath, extension=0):
        """
        Get the header from a FITS file

        Args:
            filepath (str): Path to the FITS file
            extension (int): Extension number (default: 0 for primary)

        Returns:
            astropy.io.fits.Header: The FITS header
        """
        try:
            with fits.open(filepath) as hdul:
                header = hdul[extension].header.copy()
                self.logger.info(f"Successfully read header from: {filepath}")
                return header
        except Exception as e:
            self.logger.error(f"Error reading header from {filepath}: {e}")
            return None

    def get_fits_data(self, filepath, extension=0):
        """
        Get the data from a FITS file

        Args:
            filepath (str): Path to the FITS file
            extension (int): Extension number (default: 0 for primary)

        Returns:
            numpy.ndarray: The FITS data
        """
        try:
            with fits.open(filepath) as hdul:
                data = hdul[extension].data.copy()
                self.logger.info(f"Successfully read data from: {filepath}")
                return data
        except Exception as e:
            self.logger.error(f"Error reading data from {filepath}: {e}")
            return None

    def run_pipeline(self):

        self.sort_data()

        self.create_setup_table()
