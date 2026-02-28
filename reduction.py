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

            # Open file and get HDUList (match_image_type now expects the whole HDU/HDUList)
            hdul = self.open_fits_file(filepath)
            if hdul is None:
                self.logger.warning(f"Could not open {filename} to read headers, skipping")
                continue

            try:
                image_type = self.instrument.match_image_type(hdul)
                print(f"File: {filename}, matched image type: {image_type}")
            except Exception as e:
                self.logger.warning(f"match_image_type failed for {filename}: {e}")
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

        if not getattr(self, "science_files", None):
            self.logger.info("No science files available to build setup table.")
            self.setup_table = {}
            return self.setup_table

        setup_table = {}
        key_map = {}
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
                val = header.get(filter_keyword)
                filter_names.append(str(val).strip() if val not in (None, "") else "UNKNOWN")

            filter_name = filter_names
            key = (window, bin_x, bin_y, tuple(filter_name))

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
                    "filter": filter_name,
                    "files": [file],
                }
                setup_counter += 1

            self.logger.debug(f"File: {file}, Window: {window}, Bin X: {bin_x}, Bin Y: {bin_y}, Filter: {filter_name}")

        self.logger.info(f"Created setup table with {len(setup_table)} unique setups")
        self.logger.info(f"Setup table contents: {setup_table}")

        self.setup_table = setup_table
        return self.setup_table


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


    def make_master_flats(self):

        raw_frame_dict = {}
        matched_files = 0
        total_flat_files = len(self.flat_files) if getattr(self, "flat_files", None) else 0

        for file in self.flat_files:

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
            filter_keyword_spec = self.instrument.filter_keyword[0]

            window = hdul[window_header_extension].header.get(window_keyword, "UNKNOWN")
            bin_x = hdul[bin_x_header_extension].header.get(bin_x_keyword, "UNKNOWN")
            bin_y = hdul[bin_y_header_extension].header.get(bin_y_keyword, "UNKNOWN")

            # Build filters list robustly whether filter_keyword_spec is a single key or list/tuple
            filters = []
            if isinstance(filter_keyword_spec, (list, tuple)):
                for key in filter_keyword_spec:
                    val = hdul[filter_header_extension].header.get(key)
                    filters.append(str(val).strip() if val not in (None, "") else "UNKNOWN")
            else:
                val = hdul[filter_header_extension].header.get(filter_keyword_spec)
                filters.append(str(val).strip() if val not in (None, "") else "UNKNOWN")

            key = (window, bin_x, bin_y, tuple(filters))

            if not getattr(self, "flat_configurations", None):
                self.logger.warning(f"No flat configurations available; skipping {file}")
                continue

            # Match against flat_configurations (not bias_configurations) and normalize filter types
            matched_idx = next(
                (
                    idx
                    for idx, cfg in self.flat_configurations.items()
                    if (
                        cfg.get("window"),
                        cfg.get("bin_x"),
                        cfg.get("bin_y"),
                        tuple(cfg.get("filter")) if isinstance(cfg.get("filter"), (list, tuple)) else (cfg.get("filter"),),
                    )
                    == key
                ),
                None,
            )

            if matched_idx is None:
                self.logger.warning(f"No matching flat configuration for {key} in file {file}, skipping")
                continue

            data = self.get_fits_data(file)
            if data is None:
                self.logger.warning(f"Could not read data from {file}, skipping")
                continue

            # store CCDData object under the integer configuration key; append if multiple frames
            raw_frame_dict.setdefault(matched_idx, []).append(data)
            matched_files += 1
            self.logger.info(f"Appended CCDData for {file} to raw_frame_dict[{matched_idx}] (total {len(raw_frame_dict[matched_idx])} frames)")

        # report how many flat files matched
        self.logger.info(f"Matched {matched_files} flat files out of {total_flat_files} flat files")

        self.master_flats = {}


        # first, median combine flats for each configuration
        for cfg_idx, frames in raw_frame_dict.items():
            if not frames:
                self.logger.warning(f"No frames found for configuration {cfg_idx}, skipping")
                continue

            try:
                # Use CCDData objects directly with Combiner
                comb = Combiner(frames)
                master = comb.median_combine()
                self.logger.info(f"Combined {len(frames)} CCDData frames using ccdproc.Combiner for flat {cfg_idx}")
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

            self.master_flats[cfg_idx] = master
            self.logger.info(f"Created master flat for config {cfg_idx} from {len(frames)} frames")

            # Plot each master flat
            master_data = master.data if hasattr(master, 'data') else master
            master_data_norm = master_data / np.median(master_data) if np.median(master_data) != 0 else master_data
            # set any normalized pixels outside [0.5, 1.5] to 1.0
            mask = (master_data_norm < 0.5) | (master_data_norm > 1.5)
            if np.any(mask):
                n_changed = int(np.count_nonzero(mask))
                self.logger.info(f"Master flat {cfg_idx}: {n_changed} pixels outside [0.5,1.5], setting to 1.0")
                master_data_norm = master_data_norm.copy()
                master_data_norm[mask] = 1.0
            plt.figure(figsize=(8, 6))
            plt.imshow(master_data_norm, origin="lower", cmap="gray")
            plt.title(f"Master flat {cfg_idx}")
            plt.colorbar()
            plt.show()

            try:
                master_dir = os.path.join(self.raw_data_path, "master_flats")
                os.makedirs(master_dir, exist_ok=True)

                # ensure a floating point dtype for saving
                data_to_save = master_data_norm.astype(np.float32)

                # Build header from master if present, otherwise create new one
                if hasattr(master, "header") and master.header is not None:
                    if isinstance(master.header, fits.Header):
                        header = master.header.copy()
                    else:
                        header = fits.Header()
                        try:
                            for k, v in master.header.items():
                                header[k] = v
                        except Exception:
                            # skip problematic header entries
                            pass
                else:
                    header = fits.Header()

                # Standard metadata
                header["IMTYPE"] = "MASTER_FLAT"
                header["FCFGIDX"] = cfg_idx
                header["NFRAMES"] = len(frames)
                if hasattr(master, "unit") and master.unit is not None:
                    header["BUNIT"] = str(master.unit)

                # Add flat configuration details if available
                if cfg_idx in self.flat_configurations:
                    cfg = self.flat_configurations[cfg_idx]
                    header["WINDOW"] = str(cfg.get("window", "UNKNOWN"))
                    header["BINX"] = str(cfg.get("bin_x", "UNKNOWN"))
                    header["BINY"] = str(cfg.get("bin_y", "UNKNOWN"))
                    filters = cfg.get("filter")
                    if isinstance(filters, (list, tuple)):
                        header["FILTER"] = ",".join(str(f) for f in filters)
                    else:
                        header["FILTER"] = str(filters)

                hdu = fits.PrimaryHDU(data=data_to_save, header=header)
                outname = f"master_flat_{cfg_idx:03d}.fits"
                outpath = os.path.join(master_dir, outname)
                hdu.writeto(outpath, overwrite=True)
                self.logger.info(f"Saved master flat {cfg_idx} to {outpath}")
            except Exception as e:
                self.logger.error(f"Failed to save master flat {cfg_idx}: {e}")



        
    def reduce(self):
        if not getattr(self, "setup_table", None):
            self.logger.info("No setup table available; nothing to reduce.")
            return
        # Build quick lookup maps from configurations to indices
        bias_map = {}
        if getattr(self, "bias_configurations", None):
            for idx, cfg in self.bias_configurations.items():
                bias_map[(cfg.get("window"), cfg.get("bin_x"), cfg.get("bin_y"))] = idx
        flat_map = {}
        if getattr(self, "flat_configurations", None):
            for idx, cfg in self.flat_configurations.items():
                filters = cfg.get("filter")
                if isinstance(filters, (list, tuple)):
                    filt_key = tuple(filters)
                else:
                    filt_key = (filters,)
                flat_map[(cfg.get("window"), cfg.get("bin_x"), cfg.get("bin_y"), filt_key)] = idx
        def load_master_bias(idx):
            if idx is None:
                return None
            # check in-memory
            if getattr(self, "master_biases", None) and idx in self.master_biases:
                return self.master_biases[idx]
            # try loading from disk
            path = os.path.join(self.raw_data_path, "master_biases", f"master_bias_{idx:03d}.fits")
            if os.path.exists(path):
                self.logger.info(f"Loading master bias from {path}")
                return self.get_fits_data(path)
            self.logger.warning(f"No master bias found for index {idx}")
            return None
        def load_master_flat(idx):
            if idx is None:
                return None
            if getattr(self, "master_flats", None) and idx in self.master_flats:
                return self.master_flats[idx]
            path = os.path.join(self.raw_data_path, "master_flats", f"master_flat_{idx:03d}.fits")
            if os.path.exists(path):
                self.logger.info(f"Loading master flat from {path}")
                return self.get_fits_data(path)
            self.logger.warning(f"No master flat found for index {idx}")
            return None
        reduced_dir = os.path.join(self.raw_data_path, "reduced")
        os.makedirs(reduced_dir, exist_ok=True)
        for setup_idx, setup in sorted(self.setup_table.items()):
            window = setup.get("window")
            bin_x = setup.get("bin_x")
            bin_y = setup.get("bin_y")
            filters = setup.get("filter", [])
            filt_key = tuple(filters) if isinstance(filters, (list, tuple)) else (filters,)
            bias_idx = bias_map.get((window, bin_x, bin_y))
            flat_idx = flat_map.get((window, bin_x, bin_y, filt_key))
            master_bias = load_master_bias(bias_idx)
            master_flat = load_master_flat(flat_idx)
            if master_bias is None:
                self.logger.info(f"Proceeding without bias subtraction for setup {setup_idx}")
            if master_flat is None:
                self.logger.info(f"Proceeding without flat division for setup {setup_idx}")
            for sci_file in setup.get("files", []):
                sci_ccd = self.get_fits_data(sci_file)
                if sci_ccd is None:
                    self.logger.warning(f"Could not read science file {sci_file}, skipping")
                    continue
                # keep a copy for "before" plotting
                before_data = np.array(sci_ccd.data, copy=True)
                # operate in float to avoid integer truncation
                try:
                    data = sci_ccd.data.astype(np.float32)
                except Exception as e:
                    self.logger.error(f"Could not convert science data to float for {sci_file}: {e}")
                    continue
                # Bias subtraction
                if master_bias is not None:
                    bias_data = master_bias.data if hasattr(master_bias, "data") else np.asarray(master_bias)
                    if bias_data.shape != data.shape:
                        self.logger.warning(f"Bias shape {bias_data.shape} != science shape {data.shape} for {sci_file}; skipping bias subtraction")
                    else:
                        data = data - bias_data
                        self.logger.info(f"Bias subtracted for {sci_file} using bias idx {bias_idx}")
                # Flat division
                if master_flat is not None:
                    flat_data = master_flat.data if hasattr(master_flat, "data") else np.asarray(master_flat)
                    if flat_data.shape != data.shape:
                        self.logger.warning(f"Flat shape {flat_data.shape} != science shape {data.shape} for {sci_file}; skipping flat division")
                    else:
                        # avoid division by zero or tiny numbers
                        flat_safe = flat_data.astype(np.float32).copy()
                        tiny_mask = np.isclose(flat_safe, 0.0)
                        if np.any(tiny_mask):
                            n = int(np.count_nonzero(tiny_mask))
                            self.logger.warning(f"{n} flat pixels near zero for flat idx {flat_idx}; setting them to 1.0 to avoid division by zero")
                            flat_safe[tiny_mask] = 1.0
                        data = data / flat_safe
                        self.logger.info(f"Flat divided for {sci_file} using flat idx {flat_idx}")
                # Prepare header for output
                if hasattr(sci_ccd, "header") and sci_ccd.header is not None:
                    out_header = sci_ccd.header.copy()
                else:
                    out_header = fits.Header()
                out_header["IMTYPE"] = "REDUCED"
                out_header["PROC"] = "BIAS_SUB" if (master_bias is not None and master_flat is None) else ("BIAS_SUB+FLAT_DIV" if (master_bias is not None and master_flat is not None) else ("FLAT_DIV" if (master_flat is not None) else "NONE"))
                if bias_idx is not None:
                    out_header["BCFGIDX"] = bias_idx
                if flat_idx is not None:
                    out_header["FCFGIDX"] = flat_idx
                # Save reduced file
                base = os.path.basename(sci_file)
                name_root, ext = os.path.splitext(base)
                outname = f"{name_root}_reduced{ext}"
                outpath = os.path.join(reduced_dir, outname)
                try:
                    hdu = fits.PrimaryHDU(data.astype(np.float32), header=out_header)
                    hdu.writeto(outpath, overwrite=True)
                    self.logger.info(f"Saved reduced file to {outpath}")
                except Exception as e:
                    self.logger.error(f"Failed to save reduced file {outpath}: {e}")
                # Show before and after
                try:
                    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
                    im1 = axes[0].imshow(before_data/np.median(before_data), origin="lower", cmap="gray", vmin=0.5, vmax=1.5)
                    axes[0].set_title(f"Before: {base}")
                    axes[0].axis("off")
                    fig.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)
                    im2 = axes[1].imshow(data/np.median(data), origin="lower", cmap="gray", vmin=0.5, vmax=1.5)
                    axes[1].set_title(f"After: {name_root}_reduced")
                    axes[1].axis("off")
                    fig.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)
                    plt.tight_layout()
                    plt.show()
                except Exception as e:
                    self.logger.warning(f"Failed to plot before/after for {sci_file}: {e}")

        

    def run_pipeline(self):

        self.sort_data()

        self.create_setup_table()

        self.determine_bias_configurations()

        self.make_master_bias()


        self.determine_flat_configurations()

        self.make_master_flats()

        self.reduce()
