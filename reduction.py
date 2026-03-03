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
import json


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
        self.science_to_bias_map = {}
        self.flat_configurations = {}
        self.science_to_flat_map = {}

        self.master_biases = {}
        self.bad_pixel_masks = {}

        self.master_flats = {}

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


        try:
            outpath = os.path.join(self.raw_data_path, "setup_table.json")
            serializable = {}
            for k, v in self.setup_table.items():
                serializable[k] = {
                    "window": str(v.get("window", "UNKNOWN")),
                    "bin_x": str(v.get("bin_x", "UNKNOWN")),
                    "bin_y": str(v.get("bin_y", "UNKNOWN")),
                    "filter": [str(f) for f in (v.get("filter") or [])],
                    "files": [str(f) for f in (v.get("files") or [])],
                }
            with open(outpath, "w", encoding="utf-8") as fh:
                json.dump(serializable, fh, indent=2, ensure_ascii=False)
            self.logger.info(f"Wrote setup table to {outpath}")
        except Exception as e:
            self.logger.error(f"Failed to write setup table to JSON: {e}")

        return self.setup_table



    def load_setup_table(self):

            try:
                inpath = os.path.join(self.raw_data_path, "setup_table.json")
                with open(inpath, "r", encoding="utf-8") as fh:
                    loaded = json.load(fh)
                setup_table = {}
                for k, v in loaded.items():
                    setup_table[int(k)] = {
                        "window": v.get("window", "UNKNOWN"),
                        "bin_x": v.get("bin_x", "UNKNOWN"),
                        "bin_y": v.get("bin_y", "UNKNOWN"),
                        "filter": v.get("filter", []),
                        "files": v.get("files", []),
                    }
                self.setup_table = setup_table
                self.logger.info(f"Loaded setup table from {inpath}")
            except Exception as e:
                self.logger.error(f"Failed to load setup table from JSON: {e}")
                self.setup_table = {}


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

        Also build a mapping of science setup indices -> bias configuration index
        and write that mapping to disk.

        Additionally collect bias filenames for each bias configuration (similar to setup_table).
        """

        if not getattr(self, "setup_table", None):
            self.logger.info("No setup table available to determine bias configurations.")
            self.bias_configurations = {}
            self.science_to_bias_map = {}
            return self.bias_configurations
        
        # Start by creating unique bias configurations from the setup table (so we
        # ensure science setups have corresponding bias configs where possible).
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
            unique_map[idx] = {"window": w, "bin_x": bx, "bin_y": by, "files": []}
            idx += 1

        self.bias_configurations = unique_map

        # build a quick lookup by (window, bin_x, bin_y) -> bias_idx
        bias_lookup = {(cfg["window"], cfg["bin_x"], cfg["bin_y"]): bidx for bidx, cfg in self.bias_configurations.items()}

        # Now, iterate through available bias files and append them to the proper configuration.
        bias_files = getattr(self, "bias_files", []) or []
        next_idx = max(self.bias_configurations.keys()) + 1 if self.bias_configurations else 0

        for bf in bias_files:
            hdul = self.open_fits_file(bf)
            if hdul is None:
                self.logger.warning(f"Could not open bias file {bf}, skipping")
                continue

            try:
                window_header_extension = self.instrument.detector.window_keyword[1]
                bin_x_header_extension = self.instrument.detector.bin_x_keyword[1]
                bin_y_header_extension = self.instrument.detector.bin_y_keyword[1]

                window_keyword = self.instrument.detector.window_keyword[0]
                bin_x_keyword = self.instrument.detector.bin_x_keyword[0]
                bin_y_keyword = self.instrument.detector.bin_y_keyword[0]

                w = hdul[window_header_extension].header.get(window_keyword, "UNKNOWN")
                bx = hdul[bin_x_header_extension].header.get(bin_x_keyword, "UNKNOWN")
                by = hdul[bin_y_header_extension].header.get(bin_y_keyword, "UNKNOWN")
            except Exception as e:
                self.logger.warning(f"Failed to read header keywords from bias file {bf}: {e}")
                w, bx, by = "UNKNOWN", "UNKNOWN", "UNKNOWN"

            key = (w, bx, by)
            matched_idx = bias_lookup.get(key)
            if matched_idx is None:
                # create a new bias configuration to hold this bias file
                matched_idx = next_idx
                self.bias_configurations[matched_idx] = {"window": w, "bin_x": bx, "bin_y": by, "files": [bf]}
                bias_lookup[key] = matched_idx
                next_idx += 1
                self.logger.info(f"Created new bias configuration {matched_idx} for key {key} and added file {bf}")
            else:
                files_list = self.bias_configurations[matched_idx].setdefault("files", [])
                if bf not in files_list:
                    files_list.append(bf)
                    self.logger.info(f"Appended bias file {bf} to bias configuration {matched_idx}")

        # map each science setup (setup_table key) to a bias configuration index (or None)
        science_to_bias = {}
        for setup_idx, setup in self.setup_table.items():
            key = (setup.get("window"), setup.get("bin_x"), setup.get("bin_y"))
            matched = bias_lookup.get(key)
            science_to_bias[setup_idx] = matched

        self.science_to_bias_map = science_to_bias

        self.logger.info(f"Found {len(self.bias_configurations)} unique bias configurations")
        self.logger.info(f"Bias configurations: {self.bias_configurations}")
        self.logger.info(f"Science -> Bias mapping: {self.science_to_bias_map}")

        try:
            outpath = os.path.join(self.raw_data_path, "bias_configurations.json")
            serializable = {}
            for k, v in self.bias_configurations.items():
                serializable[str(k)] = {
                    "window": str(v.get("window", "UNKNOWN")),
                    "bin_x": str(v.get("bin_x", "UNKNOWN")),
                    "bin_y": str(v.get("bin_y", "UNKNOWN")),
                    "files": [str(f) for f in (v.get("files") or [])],
                }
            with open(outpath, "w", encoding="utf-8") as fh:
                json.dump(serializable, fh, indent=2, ensure_ascii=False)
            self.logger.info(f"Wrote bias configurations to {outpath}")
        except Exception as e:
            self.logger.error(f"Failed to write bias configurations to JSON: {e}")

        try:
            map_outpath = os.path.join(self.raw_data_path, "science_to_bias_map.json")
            # ensure keys are strings for JSON compatibility and None becomes null
            serializable_map = {str(k): (v if v is not None else None) for k, v in self.science_to_bias_map.items()}
            with open(map_outpath, "w", encoding="utf-8") as fh:
                json.dump(serializable_map, fh, indent=2, ensure_ascii=False)
            self.logger.info(f"Wrote science->bias mapping to {map_outpath}")
        except Exception as e:
            self.logger.error(f"Failed to write science->bias mapping to JSON: {e}")

        return self.bias_configurations


    def load_bias_configurations(self):

        try:
            inpath = os.path.join(self.raw_data_path, "bias_configurations.json")
            with open(inpath, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            bias_configurations = {}
            for k, v in loaded.items():
                bias_configurations[int(k)] = {
                    "window": v.get("window", "UNKNOWN"),
                    "bin_x": v.get("bin_x", "UNKNOWN"),
                    "bin_y": v.get("bin_y", "UNKNOWN"),
                    "files": v.get("files", []),
                }
            self.bias_configurations = bias_configurations
            self.logger.info(f"Loaded bias configurations from {inpath}")
        except Exception as e:
            self.logger.error(f"Failed to load bias configurations from JSON: {e}")
            self.bias_configurations = {}


    def load_science_to_bias_map(self):

        try:
            inpath = os.path.join(self.raw_data_path, "science_to_bias_map.json")
            with open(inpath, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            science_to_bias_map = {}
            for k, v in loaded.items():
                science_to_bias_map[int(k)] = v if v is not None else None
            self.science_to_bias_map = science_to_bias_map
            self.logger.info(f"Loaded science->bias mapping from {inpath}")
        except Exception as e:
            self.logger.error(f"Failed to load science->bias mapping from JSON: {e}")
            self.science_to_bias_map = {}

    def make_master_bias(self):
        
        self.master_biases = {}
        self.bad_pixel_masks = {}

        for key, configuration in self.bias_configurations.items():
            self.logger.info(f"Processing bias configuration: {configuration}")

            master_bias = None
            bad_pixel_mask = None

            for bias_file in configuration.get("files", []):
                data = self.get_fits_data(bias_file)
                if data is None:
                    self.logger.warning(f"Could not read bias file {bias_file}, skipping")
                    continue

                # store CCDData objects in a list for median combining later
                if master_bias is None:
                    master_bias = [data]
                else:
                    master_bias.append(data)

            
            combiner = Combiner(master_bias)

            combined_median = combiner.median_combine()

            data = np.asarray(combined_median.data)

            # Compute statistics on finite pixels only
            finite_mask = np.isfinite(data)
            if not np.any(finite_mask):
                self.logger.warning("No finite pixels in combined median; skipping masking and histogram")
                # still store results and continue
                bad_pixel_mask = ~finite_mask
                master_bias = combined_median
                self.master_biases[str(configuration)] = master_bias
                self.bad_pixel_masks[str(configuration)] = bad_pixel_mask
            else:
                vals = data[finite_mask].ravel()

                # compute global median (used for both steps)
                median_val = float(np.median(vals))

                # --- Step 1: build a copy and apply median ±0.1*median mask ---
                if median_val == 0.0:
                    tol_small = 1e-6
                    lower_small = -tol_small
                    upper_small = tol_small
                else:
                    delta_small = 0.2 * abs(median_val)
                    lower_small = median_val - delta_small
                    upper_small = median_val + delta_small

                self.logger.info(f"Master bias initial median: {median_val:.3f}, ±0.1*median = [{lower_small:.3f}, {upper_small:.3f}]")

                # mask for the small window on a copy
                non_finite = ~finite_mask
                mask_small = non_finite | (data < lower_small) | (data > upper_small)
                data_copy = data.copy()
                masked_copy = np.ma.masked_array(data_copy, mask=mask_small)

                # get values remaining in the copy to estimate sigma
                kept_vals_small = masked_copy.compressed()
                if kept_vals_small.size == 0:
                    # fallback to using all finite vals if nothing remains after ±0.1*median
                    self.logger.warning("No pixels remain after ±0.1*median masking on copy; using all finite pixels to estimate sigma")
                    kept_vals_small = vals

                # robust std estimate (use numpy std; fallback if degenerate)
                try:
                    sigma_est = float(np.std(kept_vals_small, ddof=1)) if kept_vals_small.size > 1 else float(np.std(kept_vals_small, ddof=0))
                except Exception:
                    sigma_est = float(np.std(kept_vals_small, ddof=0))

                if sigma_est == 0.0 or not np.isfinite(sigma_est):
                    sigma_est = max(abs(median_val) * 0.01, 1e-6)  # small fallback
                    self.logger.warning(f"Estimated sigma is zero or non-finite; using fallback sigma={sigma_est:.6g}")

                self.logger.info(f"Estimated sigma from ±0.1*median-masked copy: {sigma_est:.3f}")

                # --- Step 2: mask original using median ± 5*sigma ---
                lower_5s = median_val - 5.0 * sigma_est
                upper_5s = median_val + 5.0 * sigma_est
                self.logger.info(f"Masking original combined median with median ± 5*sigma = [{lower_5s:.3f}, {upper_5s:.3f}]")

                combined_median_mask = np.zeros_like(data, dtype=bool)
                combined_median_mask |= non_finite
                out_of_range_5s = (data < lower_5s) | (data > upper_5s)
                combined_median_mask |= out_of_range_5s
                n_masked = int(np.count_nonzero(combined_median_mask))
                n_total = data.size
                self.logger.info(f"Masked {n_masked}/{n_total} pixels outside median ± 3*sigma or non-finite")

                bad_pixel_mask = combined_median_mask.copy()

                # create a masked array of the combined median for plotting/inspection
                masked_data_5s = np.ma.masked_array(data, mask=bad_pixel_mask)

                # histogram of the values used for sigma estimation (the copy)
                if kept_vals_small.size == 0:
                    self.logger.warning("No pixels available to plot histogram after ±0.1*median masking")
                else:
                    # subsample for plotting if very large
                    if kept_vals_small.size > 200_000:
                        idx = np.random.choice(kept_vals_small.size, size=200_000, replace=False)
                        vals_plot = kept_vals_small[idx]
                    else:
                        vals_plot = kept_vals_small

                    plt.figure(figsize=(8, 5))
                    plt.hist(vals_plot, bins=100, color="C0", alpha=0.8)
                    plt.axvline(median_val, color="r", linestyle="-", linewidth=2, label=f"median = {median_val:.2f}")
                    plt.axvline(lower_small, color="orange", linestyle="--", linewidth=1.5, label=f"±0.1·median bounds")
                    plt.axvline(upper_small, color="orange", linestyle="--", linewidth=1.5)
                    plt.axvline(lower_5s, color="magenta", linestyle=":", linewidth=2, label=f"-5σ = {lower_5s:.2f}")
                    plt.axvline(upper_5s, color="magenta", linestyle=":", linewidth=2, label=f"+5σ = {upper_5s:.2f}")
                    plt.title("Histogram used for sigma estimation (±0.1·median masked copy)")
                    plt.xlabel("ADU")
                    plt.ylabel("Counts")
                    plt.legend(loc="upper right")
                    # set x-limits to include median ± 5*sigma with a small margin
                    span = upper_5s - lower_5s
                    margin = span * 0.05 if span > 0 else max(abs(median_val) * 0.1, 1.0)
                    plt.xlim(lower_5s - margin, upper_5s + margin)
                    plt.tight_layout()
                    plt.show()

                # show bad-pixel mask and masked image for visual inspection (3-sigma mask)
                plt.figure(figsize=(6, 5))
                plt.imshow(bad_pixel_mask, origin="lower", cmap="gray")
                plt.title(f"Bad pixel mask (median ± 3σ) for bias config: {configuration}")
                plt.colorbar()
                plt.show()

                plt.figure(figsize=(6, 5))
                plt.imshow(masked_data_5s, origin="lower", cmap="gray")
                plt.title(f"Masked combined master bias (median ± 5σ) for config: {configuration}")
                plt.colorbar()
                plt.show()


                # store results for this configuration
                master_bias = CCDData(data, unit=u.adu, header=combined_median.header)
                self.master_biases[str(key)] = master_bias
                self.bad_pixel_masks[str(key)] = bad_pixel_mask


                try:
                    outdir = os.path.join(self.raw_data_path, "master_biases")
                    os.makedirs(outdir, exist_ok=True)
                    outpath = os.path.join(outdir, f"master_bias_{key}.fits")
                    # write only the data array, no header manipulation
                    fits.PrimaryHDU(data.astype(np.float32)).writeto(outpath, overwrite=True)
                    self.logger.info(f"Wrote master bias {key} to {outpath}")
                except Exception as e:
                    self.logger.error(f"Failed to write master bias {key} to disk: {e}")


                # save bad pixel mask to disk
                try:
                    if bad_pixel_mask is None:
                        self.logger.warning(f"No bad pixel mask for bias config {key}; skipping write")
                    else:
                        bpm_dir = os.path.join(self.raw_data_path, "bad_pixel_masks")
                        os.makedirs(bpm_dir, exist_ok=True)
                        outpath_bpm = os.path.join(bpm_dir, f"bad_pixel_mask_{key:03d}.fits")
                        # FITS doesn't always like boolean arrays; store as uint8 (0/1)
                        mask_to_write = (bad_pixel_mask.astype(np.uint8))
                        
                        # Handle header properly - convert OrderedDict to fits.Header if needed
                        if hasattr(combined_median, "header") and combined_median.header is not None:
                            if isinstance(combined_median.header, fits.Header):
                                # Already a FITS header, just copy it
                                header = combined_median.header.copy()
                            else:
                                # Convert OrderedDict or other dict-like object to FITS Header
                                header = fits.Header()
                                try:
                                    for hkey, hvalue in combined_median.header.items():
                                        header[hkey] = hvalue
                                except Exception as header_error:
                                    # Skip problematic header entries
                                    self.logger.warning(f"Could not add header key {hkey}={hvalue}: {header_error}")
                        else:
                            # Create new header if none exists
                            header = fits.Header()
                        
                        fits.PrimaryHDU(mask_to_write, header=header).writeto(outpath_bpm, overwrite=True)
                        self.logger.info(f"Wrote bad pixel mask for bias config {key} to {outpath_bpm}")
                except Exception as e:
                    self.logger.error(f"Failed to write bad pixel mask for bias config {key} to disk: {e}")


    def load_master_biases(self):

        self.master_biases = {}

        try:
            master_bias_dir = os.path.join(self.raw_data_path, "master_biases")
            if not os.path.isdir(master_bias_dir):
                self.logger.warning(f"Master bias directory {master_bias_dir} does not exist")
                return

            for filename in os.listdir(master_bias_dir):
                if not filename.lower().endswith((".fits", ".fit", ".fts")):
                    continue
                filepath = os.path.join(master_bias_dir, filename)
                hdul = self.open_fits_file(filepath)
                if hdul is None:
                    self.logger.warning(f"Could not open master bias file {filepath}, skipping")
                    continue
                data = hdul[0].data
                header = hdul[0].header
                # Extract the key properly - remove prefix and suffix, then convert to int then to str
                key_part = filename.replace("master_bias_", "").replace(".fits", "")
                try:
                    # Convert to int first to handle zero-padding, then back to str for consistency with storage
                    key_int = int(key_part)
                    key_str = str(key_int)
                    self.master_biases[key_str] = CCDData(data, unit=u.adu, header=header)
                    self.logger.info(f"Loaded master bias from {filepath} with key {key_str}")
                except ValueError as e:
                    self.logger.warning(f"Could not parse key from filename {filename}: {e}")
                    continue
        except Exception as e:
            self.logger.error(f"Failed to load master biases from disk: {e}")


    def load_bad_pixel_masks(self):

        self.bad_pixel_masks = {}

        try:
            bpm_dir = os.path.join(self.raw_data_path, "bad_pixel_masks")
            if not os.path.isdir(bpm_dir):
                self.logger.warning(f"Bad pixel mask directory {bpm_dir} does not exist")
                return

            for filename in os.listdir(bpm_dir):
                if not filename.lower().endswith((".fits", ".fit", ".fts")):
                    continue
                filepath = os.path.join(bpm_dir, filename)
                hdul = self.open_fits_file(filepath)
                if hdul is None:
                    self.logger.warning(f"Could not open bad pixel mask file {filepath}, skipping")
                    continue
                data = hdul[0].data.astype(bool)  # ensure boolean mask
                header = hdul[0].header
                # Extract the key properly - remove prefix and suffix, then convert to int then to str
                key_part = filename.replace("bad_pixel_mask_", "").replace(".fits", "")
                try:
                    # Convert to int first to handle zero-padding, then back to str for consistency with storage
                    key_int = int(key_part)
                    key_str = str(key_int)
                    self.bad_pixel_masks[key_str] = data
                    self.logger.info(f"Loaded bad pixel mask from {filepath} with key {key_str}")
                except ValueError as e:
                    self.logger.warning(f"Could not parse key from filename {filename}: {e}")
                    continue
        except Exception as e:
            self.logger.error(f"Failed to load bad pixel masks from disk: {e}")


    def determine_flat_configurations(self):

        if not getattr(self, "setup_table", None):
            self.logger.info("No setup table available to determine flat configurations.")
            self.flat_configurations = {}
            self.science_to_flat_map = {}
            return self.flat_configurations

        # Start by creating unique flat configurations from the setup table (include filter)
        unique_map = {}
        seen = set()
        idx = 0
        for entry in self.setup_table.values():
            w = entry.get("window")
            bx = entry.get("bin_x")
            by = entry.get("bin_y")
            filters = entry.get("filter") or []
            # normalize filters into a tuple for hashing
            if isinstance(filters, (list, tuple)):
                filters_tuple = tuple(filters)
            else:
                filters_tuple = (filters,)
            key = (w, bx, by, filters_tuple)
            if key in seen:
                continue
            seen.add(key)
            # include files list to be populated from available flat files
            unique_map[idx] = {"window": w, "bin_x": bx, "bin_y": by, "filter": list(filters_tuple), "files": []}
            idx += 1

        self.flat_configurations = unique_map

        # build lookup by (window, bin_x, bin_y, tuple(filter_names)) -> flat_idx
        flat_lookup = {
            (cfg["window"], cfg["bin_x"], cfg["bin_y"], tuple(cfg["filter"])): fidx
            for fidx, cfg in self.flat_configurations.items()
        }

        # Now append actual flat files to the proper configuration (creating new configs if needed)
        flat_files = getattr(self, "flat_files", []) or []
        next_idx = max(self.flat_configurations.keys()) + 1 if self.flat_configurations else 0

        for ff in flat_files:
            hdul = self.open_fits_file(ff)
            if hdul is None:
                self.logger.warning(f"Could not open flat file {ff}, skipping")
                continue

            try:
                window_header_extension = self.instrument.detector.window_keyword[1]
                bin_x_header_extension = self.instrument.detector.bin_x_keyword[1]
                bin_y_header_extension = self.instrument.detector.bin_y_keyword[1]
                filter_header_extension = self.instrument.filter_keyword[1]

                window_keyword = self.instrument.detector.window_keyword[0]
                bin_x_keyword = self.instrument.detector.bin_x_keyword[0]
                bin_y_keyword = self.instrument.detector.bin_y_keyword[0]
                filter_keyword_spec = self.instrument.filter_keyword[0]

                w = hdul[window_header_extension].header.get(window_keyword, "UNKNOWN")
                bx = hdul[bin_x_header_extension].header.get(bin_x_keyword, "UNKNOWN")
                by = hdul[bin_y_header_extension].header.get(bin_y_keyword, "UNKNOWN")
            except Exception as e:
                self.logger.warning(f"Failed to read header keywords from flat file {ff}: {e}")
                w, bx, by = "UNKNOWN", "UNKNOWN", "UNKNOWN"
                filter_keyword_spec = self.instrument.filter_keyword[0]
                filter_header_extension = self.instrument.filter_keyword[1]

            # Read filter(s) robustly (single key or list/tuple of keys)
            filter_names = []
            try:
                if isinstance(filter_keyword_spec, (list, tuple)):
                    for key in filter_keyword_spec:
                        val = hdul[filter_header_extension].header.get(key)
                        filter_names.append(str(val).strip() if val not in (None, "") else "UNKNOWN")
                else:
                    val = hdul[filter_header_extension].header.get(filter_keyword_spec)
                    filter_names.append(str(val).strip() if val not in (None, "") else "UNKNOWN")
            except Exception as e:
                self.logger.warning(f"Failed to read filter keywords from flat file {ff}: {e}")
                filter_names = ["UNKNOWN"]

            key = (w, bx, by, tuple(filter_names))
            matched_idx = flat_lookup.get(key)
            if matched_idx is None:
                # create a new flat configuration to hold this flat file
                matched_idx = next_idx
                self.flat_configurations[matched_idx] = {
                    "window": w,
                    "bin_x": bx,
                    "bin_y": by,
                    "filter": filter_names,
                    "files": [ff],
                }
                flat_lookup[key] = matched_idx
                next_idx += 1
                self.logger.info(f"Created new flat configuration {matched_idx} for key {key} and added file {ff}")
            else:
                files_list = self.flat_configurations[matched_idx].setdefault("files", [])
                if ff not in files_list:
                    files_list.append(ff)
                    self.logger.info(f"Appended flat file {ff} to flat configuration {matched_idx}")

        # map each science setup (setup_table key) to a flat configuration index (or None)
        science_to_flat = {}
        for setup_idx, setup in self.setup_table.items():
            w = setup.get("window")
            bx = setup.get("bin_x")
            by = setup.get("bin_y")
            filters = setup.get("filter") or []
            if isinstance(filters, (list, tuple)):
                filters_tuple = tuple(filters)
            else:
                filters_tuple = (filters,)
            key = (w, bx, by, filters_tuple)
            matched = flat_lookup.get(key)
            science_to_flat[setup_idx] = matched

        self.science_to_flat_map = science_to_flat

        self.logger.info(f"Found {len(self.flat_configurations)} unique flat configurations")
        self.logger.info(f"Flat configurations: {self.flat_configurations}")
        self.logger.info(f"Science -> Flat mapping: {self.science_to_flat_map}")

        # write flat configurations to disk
        try:
            outpath = os.path.join(self.raw_data_path, "flat_configurations.json")
            serializable = {}
            for k, v in self.flat_configurations.items():
                serializable[str(k)] = {
                    "window": str(v.get("window", "UNKNOWN")),
                    "bin_x": str(v.get("bin_x", "UNKNOWN")),
                    "bin_y": str(v.get("bin_y", "UNKNOWN")),
                    "filter": [str(f) for f in (v.get("filter") or [])],
                    "files": [str(f) for f in (v.get("files") or [])],
                }
            with open(outpath, "w", encoding="utf-8") as fh:
                json.dump(serializable, fh, indent=2, ensure_ascii=False)
            self.logger.info(f"Wrote flat configurations to {outpath}")
        except Exception as e:
            self.logger.error(f"Failed to write flat configurations to JSON: {e}")

        # write science -> flat mapping to disk
        try:
            map_outpath = os.path.join(self.raw_data_path, "science_to_flat_map.json")
            serializable_map = {str(k): (v if v is not None else None) for k, v in self.science_to_flat_map.items()}
            with open(map_outpath, "w", encoding="utf-8") as fh:
                json.dump(serializable_map, fh, indent=2, ensure_ascii=False)
            self.logger.info(f"Wrote science->flat mapping to {map_outpath}")
        except Exception as e:
            self.logger.error(f"Failed to write science->flat mapping to JSON: {e}")

        return self.flat_configurations


    def load_flat_configurations(self):
        
        try:
            inpath = os.path.join(self.raw_data_path, "flat_configurations.json")
            with open(inpath, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            flat_configurations = {}
            for k, v in loaded.items():
                flat_configurations[int(k)] = {
                    "window": v.get("window", "UNKNOWN"),
                    "bin_x": v.get("bin_x", "UNKNOWN"),
                    "bin_y": v.get("bin_y", "UNKNOWN"),
                    "filter": v.get("filter", []),
                    "files": v.get("files", []),
                }
            self.flat_configurations = flat_configurations
            self.logger.info(f"Loaded flat configurations from {inpath}")
        except Exception as e:
            self.logger.error(f"Failed to load flat configurations from JSON: {e}")
            self.flat_configurations = {}


    def load_science_to_flat_map(self):

        try:
            inpath = os.path.join(self.raw_data_path, "science_to_flat_map.json")
            with open(inpath, "r", encoding="utf-8") as fh:
                loaded = json.load(fh)
            science_to_flat_map = {}
            for k, v in loaded.items():
                science_to_flat_map[int(k)] = v if v is not None else None
            self.science_to_flat_map = science_to_flat_map
            self.logger.info(f"Loaded science->flat mapping from {inpath}")
        except Exception as e:
            self.logger.error(f"Failed to load science->flat mapping from JSON: {e}")
            self.science_to_flat_map = {}


    def make_master_flats(self):

        self.master_flats = {}


        for key, configuration in self.flat_configurations.items():
            self.logger.info(f"Processing flat configuration: {configuration}")

            master_flat = None

            for flat_file in configuration.get("files", []):
                data = self.get_fits_data(flat_file)
                if data is None:
                    self.logger.warning(f"Could not read flat file {flat_file}, skipping")
                    continue

                if master_flat is None:
                    master_flat = [data]        
                else:
                    master_flat.append(data)

            if master_flat is not None:
                combiner = Combiner(master_flat)
                combined_median = combiner.median_combine()

                # choose bias using science->flat and science->bias mappings:
                bias_frame = None
                bad_pixel_mask = None


                science_conf = self.science_to_flat_map.get(key)
                bias_conf_idx = self.science_to_bias_map.get(science_conf)

                if bias_conf_idx is None:
                    continue

                bias_frame = self.master_biases.get(str(bias_conf_idx))
                bad_pixel_mask = self.bad_pixel_masks.get(str(bias_conf_idx))

                print(bad_pixel_mask)


                if bad_pixel_mask is None:
                    self.logger.error(f"No bad pixel mask found for bias config idx {bias_conf_idx} used in flat config {key}")
                    exit(-1)


                masked_bias = np.ma.masked_array(bias_frame.data, mask=bad_pixel_mask)

                #TODO: for debugging:
                if bias_frame is None:
                    self.logger.error("NO BIAS FRAME - ABORTING")
                    exit(-1)

                plt.imshow(masked_bias, origin="lower", cmap="gray")
                plt.title(f"Bias frame for bias config idx {bias_conf_idx} used in flat config {key}")
                plt.colorbar()
                plt.show()

                masked_bias = np.ma.masked_array(bias_frame.data, mask=bad_pixel_mask) 

                masked_flat = np.ma.masked_array(combined_median.data, mask=bad_pixel_mask) 

                bias_subtracted_flat = masked_flat - masked_bias

                work_arr = np.asarray(bias_subtracted_flat)
                finite_mask = np.isfinite(work_arr)

                vals = work_arr[finite_mask].ravel()
                median_val = float(np.median(vals))

                delta_small = 0.2 * abs(median_val)
                lower_small = median_val - delta_small
                upper_small = median_val + delta_small

                non_finite = ~finite_mask
                mask_small = non_finite | (work_arr < lower_small) | (work_arr > upper_small)
                data_copy = work_arr.copy()
                masked_copy = np.ma.masked_array(data_copy, mask=mask_small)

                kept_vals_small = masked_copy.compressed()

                try:
                    sigma_est = float(np.std(kept_vals_small, ddof=1))
                except Exception:
                    sigma_est = float(np.std(kept_vals_small, ddof=0))

                try:
                    fallback = max(abs(median_val) * 0.01, 1e-6)
                except Exception:
                    fallback = 1e-6

                sigma_est = np.nan_to_num(sigma_est, nan=fallback)

                mean_estimate = float(np.mean(kept_vals_small)) if kept_vals_small.size > 0 else median_val

                lower_5s = median_val - 5.0 * sigma_est
                upper_5s = median_val + 5.0 * sigma_est

                combined_mask = np.zeros_like(work_arr, dtype=bool)
                combined_mask |= non_finite
                out_of_range_5s = (work_arr < lower_5s) | (work_arr > upper_5s)
                combined_mask |= out_of_range_5s
                n_masked = int(np.count_nonzero(combined_mask))
                n_total = work_arr.size

                bad_pixel_mask = combined_mask.copy()
                masked_data_5s = np.ma.masked_array(work_arr, mask=bad_pixel_mask)

                mean = float(np.mean(masked_data_5s)) if masked_data_5s.size > 0 else median_val

                normalized_flat = combined_median.data / mean

                normalized_flat_masked = np.ma.masked_array(normalized_flat, mask=bad_pixel_mask)

                try:
                    size_plot = min(kept_vals_small.size, 200_000)
                    idxs = np.random.choice(kept_vals_small.size, size=size_plot, replace=False)
                    vals_plot = kept_vals_small[idxs]
                except Exception:
                    vals_plot = kept_vals_small

                try:
                    plt.figure(figsize=(8, 5))
                    plt.hist(vals_plot, bins=100, color="C0", alpha=0.8)
                    plt.axvline(median_val, color="r", linestyle="-", linewidth=2, label=f"median = {median_val:.2f}")
                    plt.axvline(lower_small, color="orange", linestyle="--", linewidth=1.5, label=f"±0.2·median bounds")
                    plt.axvline(upper_small, color="orange", linestyle="--", linewidth=1.5)
                    plt.axvline(lower_5s, color="magenta", linestyle=":", linewidth=2, label=f"-5σ = {lower_5s:.2f}")
                    plt.axvline(upper_5s, color="magenta", linestyle=":", linewidth=2, label=f"+5σ = {upper_5s:.2f}")
                    plt.title("Histogram used for sigma estimation (±0.2·median masked copy)")
                    plt.xlabel("ADU")
                    plt.ylabel("Counts")
                    plt.legend(loc="upper right")
                    span = upper_5s - lower_5s
                    margin = span * 0.05 if span > 0 else max(abs(median_val) * 0.1, 1.0)
                    plt.xlim(lower_5s - margin, upper_5s + margin)
                    plt.tight_layout()
                    plt.show()
                except Exception:
                    pass

                try:
                    plt.figure(figsize=(6, 5))
                    plt.imshow(normalized_flat_masked, origin="lower", cmap="gray")
                    plt.title(f"Masked combined master flat (median ± 5σ) for config idx: {key}")
                    plt.colorbar()
                    plt.show()
                except Exception:
                    pass


                master_flat = CCDData(normalized_flat, unit=u.adu, header=combined_median.header)

                self.master_flats[key] = master_flat
                self.bad_pixel_masks[str(bias_conf_idx)] = bad_pixel_mask

                try:
                    outdir = os.path.join(self.raw_data_path, "master_flats")
                    os.makedirs(outdir, exist_ok=True)
                    outpath = os.path.join(outdir, f"master_flat_{key:03d}.fits")
                    fits.PrimaryHDU(master_flat.data.astype(np.float32)).writeto(outpath, overwrite=True)
                    self.logger.info(f"Wrote master flat {key} to {outpath}")
                except Exception as e:
                    self.logger.error(f"Failed to write master flat {key} to disk: {e}")


                # write bad pixel mask for this flat's associated bias configuration
                try:
                    if bad_pixel_mask is None:
                        self.logger.warning(f"No bad pixel mask to write for flat config {key}; skipping BPM write")
                    else:
                        bpm_dir = os.path.join(self.raw_data_path, "bad_pixel_masks")
                        os.makedirs(bpm_dir, exist_ok=True)

                        # prefer using the bias configuration index for the mask filename when available,
                        # otherwise fall back to the flat configuration key
                        bpm_idx = bias_conf_idx if bias_conf_idx is not None else key
                        if isinstance(bpm_idx, int):
                            bpm_name = f"bad_pixel_mask_{bpm_idx:03d}.fits"
                        else:
                            # sanitize non-integer keys into a string safe for filenames
                            safe = str(bpm_idx).replace(" ", "_").replace("/", "_")
                            bpm_name = f"bad_pixel_mask_{safe}.fits"

                        outpath_bpm = os.path.join(bpm_dir, bpm_name)

                        # FITS doesn't always like boolean arrays; store as uint8 (0/1)
                        mask_to_write = bad_pixel_mask.astype(np.uint8)

                        # build a FITS header from the combined median header if possible
                        if hasattr(combined_median, "header") and combined_median.header is not None:
                            if isinstance(combined_median.header, fits.Header):
                                header = combined_median.header.copy()
                            else:
                                header = fits.Header()
                                try:
                                    for hkey, hvalue in combined_median.header.items():
                                        header[hkey] = hvalue
                                except Exception:
                                    # ignore problematic header entries
                                    pass
                        else:
                            header = fits.Header()

                        fits.PrimaryHDU(mask_to_write, header=header).writeto(outpath_bpm, overwrite=True)
                        self.logger.info(f"Wrote bad pixel mask for flat config {key} to {outpath_bpm}")
                except Exception as e:
                    self.logger.error(f"Failed to write bad pixel mask for flat config {key} to disk: {e}")

    def load_master_flats(self):

        self.master_flats = {}

        try:
            master_flat_dir = os.path.join(self.raw_data_path, "master_flats")
            if not os.path.isdir(master_flat_dir):
                self.logger.warning(f"Master flat directory {master_flat_dir} does not exist")
                return

            for filename in os.listdir(master_flat_dir):
                if not filename.lower().endswith((".fits", ".fit", ".fts")):
                    continue
                filepath = os.path.join(master_flat_dir, filename)
                hdul = self.open_fits_file(filepath)
                if hdul is None:
                    self.logger.warning(f"Could not open master flat file {filepath}, skipping")
                    continue
                data = hdul[0].data
                header = hdul[0].header
                # Extract the key properly - remove prefix and suffix, then convert to int then to str
                key_part = filename.replace("master_flat_", "").replace(".fits", "")
                try:
                    # Convert to int first to handle zero-padding, then back to str for consistency with storage
                    key_int = int(key_part)
                    key_str = str(key_int)
                    self.master_flats[key_str] = CCDData(data, unit=u.adu, header=header)
                    self.logger.info(f"Loaded master flat from {filepath} with key {key_str}")
                except ValueError as e:
                    self.logger.warning(f"Could not parse key from filename {filename}: {e}")
                    continue
        except Exception as e:
            self.logger.error(f"Failed to load master flats from disk: {e}")


        
    def reduce(self):

        self.logger.info("Starting reduction process...")

        for key, configuration in self.setup_table.items():
            print(f"Reducing science frame with setup config idx {key}: {configuration}")


            bias_frame = self.master_biases.get(str(self.science_to_bias_map.get(key)))
            flat_frame = self.master_flats.get(str(self.science_to_flat_map.get(key)))
            bad_pixel_mask = self.bad_pixel_masks.get(str(self.science_to_bias_map.get(key)))

            for file in configuration.get("files", []):
                self.logger.info(f"Reducing science file {file} with setup config idx {key}")

                science_data = self.get_fits_data(file)
                if science_data is None:
                    self.logger.warning(f"Could not read science file {file}, skipping")
                    continue

                if bias_frame is None:
                    self.logger.error(f"No master bias found for science config idx {key}, skipping reduction for file {file}")
                    continue

                if flat_frame is None:
                    self.logger.error(f"No master flat found for science config idx {key}, skipping reduction for file {file}")
                    continue

                if bad_pixel_mask is None:
                    self.logger.error(f"No bad pixel mask found for science config idx {key}, skipping reduction for file {file}")
                    continue

                
                masked_science = np.ma.masked_array(science_data, mask=bad_pixel_mask)
                masked_bias = np.ma.masked_array(bias_frame.data, mask=bad_pixel_mask)
                masked_flat = np.ma.masked_array(flat_frame.data, mask=bad_pixel_mask)

                bias_subtracted = masked_science - masked_bias
                flat_corrected = bias_subtracted / masked_flat


                plt.imshow(flat_corrected/np.mean(flat_corrected), origin="lower", cmap="gray", vmin=0.5, vmax=1.5)
                plt.title(f"{file} reduced with science config idx {key}")
                plt.colorbar()
                plt.show()


    def run_pipeline(self):

        if False:

            self.sort_data()

            self.create_setup_table()

            self.determine_bias_configurations()

            self.make_master_bias()

            self.determine_flat_configurations()

            self.make_master_flats()

            self.make_master_flats()


        if True:

            self.load_bias_configurations()

            self.load_setup_table()

            self.load_science_to_bias_map()

            self.load_master_biases()

            self.load_bad_pixel_masks()

            self.load_flat_configurations()

            self.load_science_to_flat_map()

            self.load_master_flats()

            self.reduce()

        #self.reduce()
