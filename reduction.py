from instruments import Instrument
import os
from logger import init_logger
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

from instruments import ImageType
import math
from ccdproc import Combiner, CCDData
from astropy import units as u


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
        self.bias_configurations = {}
        self.flat_configurations = {}

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

    def get_fits_data(self, filepath):
        """
        Get the data from a FITS file as a CCDData object

        Args:
            filepath (str): Path to the FITS file

        Returns:
            ccdproc.CCDData: The FITS data as CCDData object with header and units
        """

        extension = self.instrument.data_hdu_extension if self.instrument.data_hdu_extension is not None else 0

        try:
            with fits.open(filepath) as hdul:
                data = hdul[extension].data
                header = hdul[extension].header
                
                # Create CCDData object with units (assuming ADU for astronomical data)
                ccd_data = CCDData(data, unit=u.adu, header=header)
                self.logger.info(f"Successfully read data from: {filepath} as CCDData")
                return ccd_data
        except Exception as e:
            self.logger.error(f"Error reading data from {filepath}: {e}")
            return None
        

    def determine_bias_configurations(self):
        
        """
        Build a new dict of unique detector configurations from self.setup_table,
        keeping only the 'window', 'bin_x' and 'bin_y' entries.
        """

        if not getattr(self, "setup_table", None):
            self.logger.info("No setup table available to determine bias configurations.")
            self.bias_configurations = {}
            return self.bias_configurations
        
        unique_map = {}
        seen = set()
        idx = 0
        for entry in self.setup_table.values():
            w = entry.get("window")
            bx = entry.get("bin_x")
            by = entry.get("bin_y")
            key = (w, bx, by)
            if key in seen:
                continue
            seen.add(key)
            unique_map[idx] = {"window": w, "bin_x": bx, "bin_y": by}
            idx += 1

        self.bias_configurations = unique_map

        self.logger.info(f"Found {len(unique_map)} unique bias configurations")
        self.logger.info(f"Bias configurations: {self.bias_configurations}")


    def make_master_bias(self):
        
        raw_frame_dict = {}

        for file in self.bias_files:

            hdul = self.open_fits_file(file)

            if hdul is None:
                self.logger.warning(f"Could not open {file}, skipping")
                continue

            window_header_extension = self.instrument.detector.window_keyword[1]
            bin_x_header_extension = self.instrument.detector.bin_x_keyword[1]
            bin_y_header_extension = self.instrument.detector.bin_y_keyword[1]

            window_keyword = self.instrument.detector.window_keyword[0]
            bin_x_keyword = self.instrument.detector.bin_x_keyword[0]
            bin_y_keyword = self.instrument.detector.bin_y_keyword[0]

            window = hdul[window_header_extension].header.get(window_keyword, "UNKNOWN")
            bin_x = hdul[bin_x_header_extension].header.get(bin_x_keyword, "UNKNOWN")
            bin_y = hdul[bin_y_header_extension].header.get(bin_y_keyword, "UNKNOWN")

            key = (window, bin_x, bin_y)

            if not getattr(self, "bias_configurations", None):
                self.logger.warning(f"No bias configurations available; skipping {file}")
                continue

            matched_idx = next(
                (
                    idx
                    for idx, cfg in self.bias_configurations.items()
                    if (cfg.get("window"), cfg.get("bin_x"), cfg.get("bin_y")) == key
                ),
                None,
            )

            if matched_idx is None:
                self.logger.warning(f"No matching bias configuration for {key} in file {file}, skipping")
                continue

            data = self.get_fits_data(file)
            if data is None:
                self.logger.warning(f"Could not read data from {file}, skipping")
                continue

            # store CCDData object under the integer configuration key; append if multiple frames
            raw_frame_dict.setdefault(matched_idx, []).append(data)
            self.logger.info(f"Appended CCDData for {file} to raw_frame_dict[{matched_idx}] (total {len(raw_frame_dict[matched_idx])} frames)")


        self.master_biases = {}

        if not raw_frame_dict:
            self.logger.info("No bias frames to combine.")
            return

        for cfg_idx, frames in raw_frame_dict.items():
            if not frames:
                self.logger.warning(f"No frames for configuration {cfg_idx}, skipping")
                continue

            try:
                # Use CCDData objects directly with Combiner
                comb = Combiner(frames)
                master = comb.median_combine()
                self.logger.info(f"Combined {len(frames)} CCDData frames using ccdproc.Combiner with sigma clipping")
            except Exception as e:
                # Fallback to numpy median if Combiner fails
                self.logger.warning(f"ccdproc.Combiner failed for config {cfg_idx}: {e}. Falling back to numpy.median")
                try:
                    # Extract data arrays from CCDData objects for numpy fallback
                    data_arrays = [frame.data for frame in frames]
                    stacked = np.stack(data_arrays, axis=0)
                    median_data = np.median(stacked, axis=0)
                    # Create a new CCDData object with the combined data
                    master = CCDData(median_data, unit=frames[0].unit, header=frames[0].header)
                except Exception as fallback_error:
                    self.logger.error(f"Both ccdproc and numpy fallback failed for config {cfg_idx}: {fallback_error}")
                    continue

            self.master_biases[cfg_idx] = master
            self.logger.info(f"Created master bias for config {cfg_idx} from {len(frames)} frames")

        # Plot the masters
        n_masters = len(self.master_biases)
        if n_masters == 0:
            self.logger.info("No master bias frames to plot.")
        else:
            ncols = min(3, n_masters)
            nrows = math.ceil(n_masters / ncols)
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
            # Normalize axes array shape for consistent indexing
            if isinstance(axes, np.ndarray):
                axes_flat = axes.flatten()
            else:
                axes_flat = [axes]

            # Turn off any unused axes
            for ax in axes_flat[n_masters:]:
                ax.axis("off")

            for i, (cfg_idx, master) in enumerate(sorted(self.master_biases.items())):
                ax = axes_flat[i]
                # Extract data from CCDData object for plotting
                master_data = master.data if hasattr(master, 'data') else master
                im = ax.imshow(master_data, origin="lower", cmap="gray")
                ax.set_title(f"Master bias {cfg_idx}")
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

            plt.tight_layout()
            plt.show()

        # save master biases to a subdirectory inside the raw data directory
        master_dir = os.path.join(self.raw_data_path, "master_biases")
        os.makedirs(master_dir, exist_ok=True)

        for cfg_idx, master in sorted(self.master_biases.items()):
            try:
                # extract data and header (support CCDData or raw numpy arrays)
                data = master.data if hasattr(master, "data") else np.asarray(master)
                
                # Handle header properly - convert OrderedDict to fits.Header if needed
                if hasattr(master, "header") and master.header is not None:
                    if isinstance(master.header, fits.Header):
                        # Already a FITS header, just copy it
                        header = master.header.copy()
                    else:
                        # Convert OrderedDict or other dict-like object to FITS Header
                        header = fits.Header()
                        for key, value in master.header.items():
                            try:
                                header[key] = value
                            except Exception as header_error:
                                # Skip problematic header entries
                                self.logger.warning(f"Could not add header key {key}={value}: {header_error}")
                else:
                    # Create new header if none exists
                    header = fits.Header()

                # record some metadata
                header["IMTYPE"] = "MASTER_BIAS"
                header["BCFGIDX"] = cfg_idx
                if hasattr(master, "unit") and master.unit is not None:
                    header["BUNIT"] = str(master.unit)

                # Add bias configuration info to header
                if cfg_idx in self.bias_configurations:
                    cfg = self.bias_configurations[cfg_idx]
                    header["WINDOW"] = str(cfg.get("window", "UNKNOWN"))
                    header["BINX"] = str(cfg.get("bin_x", "UNKNOWN"))
                    header["BINY"] = str(cfg.get("bin_y", "UNKNOWN"))

                hdu = fits.PrimaryHDU(data=data, header=header)
                outname = f"master_bias_{cfg_idx:03d}.fits"
                outpath = os.path.join(master_dir, outname)
                hdu.writeto(outpath, overwrite=True)

                self.logger.info(f"Saved master bias {cfg_idx} to {outpath}")
            except Exception as e:
                self.logger.error(f"Failed to save master bias {cfg_idx}: {e}")
                # Add more detailed error information
                self.logger.error(f"Master type: {type(master)}, has header: {hasattr(master, 'header')}")
                if hasattr(master, 'header'):
                    self.logger.error(f"Header type: {type(master.header)}")


    def determine_flat_configurations(self):

        if not getattr(self, "setup_table", None):
            self.logger.info("No setup table available to determine flat configurations.")
            self.flat_configurations = {}
            return self.flat_configurations

        unique_map = {}
        seen = set()
        idx = 0
        for entry in self.setup_table.values():
            w = entry.get("window")
            bx = entry.get("bin_x")
            by = entry.get("bin_y")
            filters = entry.get("filter")
            # Convert list to tuple to make it hashable for use in set
            filters_tuple = tuple(filters) if isinstance(filters, (list, tuple)) else (filters,)
            key = (w, bx, by, filters_tuple)
            if key in seen:
                continue
            seen.add(key)
            unique_map[idx] = {"window": w, "bin_x": bx, "bin_y": by, "filter": filters}
            idx += 1

        self.flat_configurations = unique_map

        self.logger.info(f"Found {len(unique_map)} unique flat configurations")
        self.logger.info(f"Flat configurations: {self.flat_configurations}")

    def run_pipeline(self):

        self.sort_data()

        self.create_setup_table()

        #self.determine_bias_configurations()

        #self.make_master_bias()


        self.determine_flat_configurations()
