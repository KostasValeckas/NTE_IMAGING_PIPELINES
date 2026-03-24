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
import matplotlib


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
        self.bad_pixel_masks_bias = {}

        self.master_flats = {}
        self.bad_pixel_masks_flats = {}


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

            window = self.instrument.get_header_value(hdul, self.instrument.detector.window_keyword) or "UNKNOWN"
            bin_x = self.instrument.get_header_value(hdul, self.instrument.detector.bin_x_keyword) or "UNKNOWN"
            bin_y = self.instrument.get_header_value(hdul, self.instrument.detector.bin_y_keyword) or "UNKNOWN"

            filter_names = self.instrument.get_header_values(hdul, self.instrument.filter_keyword) or ["UNKNOWN"]

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
        #TODO: HACKED - fix later
        if self.instrument.name == "NOTCAM": return

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
                # Use the instrument helper to get header values (handles string, tuple or list specs)
                try:
                    w = self.instrument.get_header_value(hdul, self.instrument.detector.window_keyword) or "UNKNOWN"
                except Exception as e:
                    self.logger.warning(f"Failed to read window keyword using get_header_value for bias file {bf}: {e}")
                    w = "UNKNOWN"

                try:
                    bx = self.instrument.get_header_value(hdul, self.instrument.detector.bin_x_keyword) or "UNKNOWN"
                except Exception as e:
                    self.logger.warning(f"Failed to read bin_x keyword using get_header_value for bias file {bf}: {e}")
                    bx = "UNKNOWN"

                try:
                    by = self.instrument.get_header_value(hdul, self.instrument.detector.bin_y_keyword) or "UNKNOWN"
                except Exception as e:
                    self.logger.warning(f"Failed to read bin_y keyword using get_header_value for bias file {bf}: {e}")
                    by = "UNKNOWN"

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
        
        #TODO: HACKED - fix later
        if self.instrument.name == "NOTCAM": return

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

        #TODO: HACKED - fix later
        if self.instrument.name == "NOTCAM": return

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

        #TODO: HACKED - fix later
        if self.instrument.name == "NOTCAM": return

        self.master_biases = {}
        self.bad_pixel_masks_bias = {}

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
                self.bad_pixel_masks_bias[str(key)] = bad_pixel_mask


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
                        bpm_dir = os.path.join(self.raw_data_path, "master_biases", "bad_pixel_masks")
                        os.makedirs(bpm_dir, exist_ok=True)
                        outpath_bpm = os.path.join(bpm_dir, f"bad_pixel_mask_bias_{key:03d}.fits")
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

        if self.instrument.name == "NOTCAM": return

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


    def load_bad_pixel_masks_bias(self):

        #TODO: HACKED - fix later
        if self.instrument.name == "NOTCAM": return

        self.bad_pixel_masks_bias = {}

        try:
            bpm_dir = os.path.join(self.raw_data_path, "master_biases", "bad_pixel_masks")
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
                    self.bad_pixel_masks_bias[key_str] = data
                    self.logger.info(f"Loaded bad pixel mask from {filepath} with key {key_str}")
                except ValueError as e:
                    self.logger.warning(f"Could not parse key from filename {filename}: {e}")
                    continue
        except Exception as e:
            self.logger.error(f"Failed to load bad pixel masks from disk: {e}")


    def load_bad_pixel_masks_flats(self):

        self.bad_pixel_masks_flats = {}

        try:
            bpm_dir = os.path.join(self.raw_data_path, "master_biases", "bad_pixel_masks")
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
                key_part = filename.replace("bad_pixel_mask_bias_", "").replace(".fits", "")
                try:
                    # Convert to int first to handle zero-padding, then back to str for consistency with storage
                    key_int = int(key_part)
                    key_str = str(key_int)
                    self.bad_pixel_masks_flats[key_str] = data
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


            # Use the instrument helper to read header values robustly
            try:
                w = self.instrument.get_header_value(hdul, self.instrument.detector.window_keyword) or "UNKNOWN"
            except Exception as e:
                self.logger.warning(f"Failed to read window keyword using get_header_value for flat file {ff}: {e}")
                w = "UNKNOWN"
            try:
                bx = self.instrument.get_header_value(hdul, self.instrument.detector.bin_x_keyword) or "UNKNOWN"
            except Exception as e:
                self.logger.warning(f"Failed to read bin_x keyword using get_header_value for flat file {ff}: {e}")
                bx = "UNKNOWN"
            try:
                by = self.instrument.get_header_value(hdul, self.instrument.detector.bin_y_keyword) or "UNKNOWN"
            except Exception as e:
                self.logger.warning(f"Failed to read bin_y keyword using get_header_value for flat file {ff}: {e}")
                by = "UNKNOWN"

            # Read filter(s) using the instrument helper which handles single or multiple keywords
            try:
                filter_names = self.instrument.get_header_values(hdul, self.instrument.filter_keyword) or ["UNKNOWN"]
            except Exception as e:
                self.logger.warning(f"Failed to read filter keywords using get_header_values for flat file {ff}: {e}")
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

                if self.instrument.name != "NOTCAM":

                    # choose bias using science->flat and science->bias mappings:
                    bias_frame = None
                    bad_pixel_mask = None


                    science_conf = self.science_to_flat_map.get(key)
                    bias_conf_idx = self.science_to_bias_map.get(science_conf)

                    if bias_conf_idx is None:
                        continue

                    bias_frame = self.master_biases.get(str(bias_conf_idx))
                    bad_pixel_mask_bias = self.bad_pixel_masks_bias.get(str(bias_conf_idx))



                    if bad_pixel_mask_bias is None:
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

                else:
                    bias_subtracted_flat = combined_median.data

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

                bad_pixel_mask_flat = combined_mask.copy()
                masked_data_5s = np.ma.masked_array(work_arr, mask=bad_pixel_mask_flat)

                mean = float(np.mean(masked_data_5s)) if masked_data_5s.size > 0 else median_val

                normalized_flat = combined_median.data / mean

                normalized_flat_masked = np.ma.masked_array(normalized_flat, mask=bad_pixel_mask_flat)

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
                self.bad_pixel_masks_flats[key] = bad_pixel_mask_flat

                try:
                    outdir = os.path.join(self.raw_data_path, "master_flats")
                    os.makedirs(outdir, exist_ok=True)
                    outpath = os.path.join(outdir, f"master_flat_{key:03d}.fits")
                    fits.PrimaryHDU(master_flat.data.astype(np.float32)).writeto(outpath, overwrite=True)
                    self.logger.info(f"Wrote master flat {key} to {outpath}")
                except Exception as e:
                    self.logger.error(f"Failed to write master flat {key} to disk: {e}")


                # write bad pixel mask for this flat configuration
                try:
                    if bad_pixel_mask_flat is None:
                        self.logger.warning(f"No bad pixel mask to write for flat config {key}; skipping BPM write")
                    else:
                        bpm_dir = os.path.join(self.raw_data_path, "master_flats", "bad_pixel_masks")
                        os.makedirs(bpm_dir, exist_ok=True)

                        # use the flat configuration index for the mask filename
                        bpm_idx = key
                        if isinstance(bpm_idx, int):
                            bpm_name = f"bad_pixel_mask_flat_{bpm_idx:03d}.fits"
                        else:
                            # sanitize non-integer keys into a string safe for filenames
                            safe = str(bpm_idx).replace(" ", "_").replace("/", "_")
                            bpm_name = f"bad_pixel_mask_flat_{safe}.fits"

                        outpath_bpm = os.path.join(bpm_dir, bpm_name)

                        # FITS doesn't always like boolean arrays; store as uint8 (0/1)
                        mask_to_write = bad_pixel_mask_flat.astype(np.uint8)

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


    def load_bad_pixel_masks_flats(self):

        self.bad_pixel_masks_flats = {}

        try:
            bpm_dir = os.path.join(self.raw_data_path, "master_flats", "bad_pixel_masks")
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
                key_part = filename.replace("bad_pixel_mask_flat_", "").replace(".fits", "")
                print(f"{filename} -> key part: {key_part}")
                try:
                    # Convert to int first to handle zero-padding, then back to str for consistency with storage
                    key_int = int(key_part)
                    key_str = str(key_int)
                    self.bad_pixel_masks_flats[key_str] = data
                    self.logger.info(f"Loaded bad pixel mask from {filepath} with key {key_str}")
                except ValueError as e:
                    self.logger.warning(f"Could not parse key from filename {filename}: {e}")
                    continue
        except Exception as e:
            self.logger.error(f"Failed to load bad pixel masks from disk: {e}")



        
    def reduce(self):

        self.logger.info("Starting reduction process...")

        for key, configuration in self.setup_table.items():
            print(f"Reducing science frame with setup config idx {key}: {configuration}")

            #if key == 0: continue


            # determine bias/flat indices and frames
            bias_idx = self.science_to_bias_map.get(key)
            bias_frame = self.master_biases.get(str(bias_idx)) if bias_idx is not None else None
            flat_idx = self.science_to_flat_map.get(key)
            flat_frame = self.master_flats.get(str(flat_idx))
            bad_pixel_mask = self.bad_pixel_masks_flats.get(str(flat_idx))


            # Build/extend a mapping of object name -> list of filenames for the current setup entries
            if not hasattr(self, "object_to_files"):
                self.object_to_files = {}

            if not hasattr(self, "reduced_objects"):
                self.reduced_objects = {}

            for fname in configuration.get("files", []):
                hdul_tmp = self.open_fits_file(fname)
                if hdul_tmp is None:
                    self.logger.warning(f"Could not open {fname} to read OBJECT header, skipping for object mapping")
                    continue
                try:
                    obj = self.instrument.get_header_value(hdul_tmp, self.instrument.object_keyword) or "UNKNOWN"
                except Exception as e:
                    self.logger.warning(f"Failed to read object name from header for {fname}: {e}")
                    obj = "UNKNOWN"
                key = str(obj)
                lst = self.object_to_files.setdefault(key, [])
                if fname not in lst:
                    lst.append(fname)


            # optional quick log
            self.logger.info(f"Updated object->files map; total objects tracked: {len(self.object_to_files)}")


            for file in configuration.get("files", []):


                self.logger.info(f"Reducing science file {file} with setup config idx {key}")
                
                hdul = self.open_fits_file(file)
                science_data = self.get_fits_data(file)
                
                if science_data is None:
                    self.logger.warning(f"Could not read science file {file}, skipping")
                    continue

                # Print name of the object being reduced if available
                try:
                    obj_name = self.instrument.get_header_value(hdul, self.instrument.object_keyword) or "UNKNOWN"
                    self.logger.info(f"Object name from header: {obj_name}")
                except Exception as e:
                    self.logger.warning(f"Failed to read object name from header for file {file}: {e}")



                if bias_frame is None and self.instrument.name != "NOTCAM":
                    self.logger.error(f"No master bias found for science config idx {key}, skipping reduction for file {file}")
                    continue

                if flat_frame is None:
                    self.logger.error(f"No master flat found for science config idx {key}, skipping reduction for file {file}")
                    continue

                if bad_pixel_mask is None:
                    self.logger.error(f"No bad pixel mask found for science config idx {key}, skipping reduction for file {file}")
                    continue

                
                # create masked arrays for science / bias / flat using the bad-pixel mask
                science_arr = science_data.data if hasattr(science_data, "data") else np.asarray(science_data)
                masked_science = np.ma.masked_array(science_arr, mask=bad_pixel_mask)

                has_bias = (bias_frame is not None) and (self.instrument.name != "NOTCAM")
                masked_bias = np.ma.masked_array(bias_frame.data, mask=bad_pixel_mask) if has_bias else None
                masked_flat = np.ma.masked_array(flat_frame.data, mask=bad_pixel_mask)

                if has_bias:
                    bias_subtracted = masked_science - masked_bias
                else:
                    bias_subtracted = masked_science 
                flat_corrected = bias_subtracted / masked_flat 


                # reduced result (masked array)
                reduced_masked = flat_corrected

                # safe percentile helper
                def safe_percentiles(values, low=5, high=95):
                    try:
                        if values is None or values.size == 0:
                            return (0.0, 1.0)
                        return (np.percentile(values, low), np.percentile(values, high))
                    except Exception:
                        mn = float(np.nanmin(values)) if values.size > 0 else 0.0
                        mx = float(np.nanmax(values)) if values.size > 0 else mn + 1.0
                        return (mn, mx)

                # compute display limits for each panel
                sci_vals = science_arr[~bad_pixel_mask] if np.any(~bad_pixel_mask) else science_arr.ravel()
                smin, smax = safe_percentiles(sci_vals)

                flat_vals = masked_flat.compressed() if hasattr(masked_flat, "compressed") else np.asarray(masked_flat).ravel()
                fmin, fmax = safe_percentiles(flat_vals)

                red_vals = reduced_masked.compressed() if hasattr(reduced_masked, "compressed") else np.asarray(reduced_masked).ravel()
                rmin, rmax = safe_percentiles(red_vals)

                if has_bias:
                    bias_vals = masked_bias.compressed() if hasattr(masked_bias, "compressed") else np.asarray(masked_bias).ravel()
                    bmin, bmax = safe_percentiles(bias_vals)

                if True:

                    # create subplots: 2x2 if bias exists, else 1x3
                    if has_bias:
                        fig, axs = plt.subplots(2, 2, figsize=(16, 10))
                        ax_list = list(axs.flatten())
                    else:
                        fig, axs = plt.subplots(1, 3, figsize=(24, 6))
                        ax_list = list(axs)

                    # Original science
                    im0 = ax_list[0].imshow(science_arr, origin="lower", cmap="gray", vmin=smin, vmax=smax)
                    ax_list[0].set_title(f"Original science frame (config {key})")
                    fig.colorbar(im0, ax=ax_list[0])

                    # Bias frame (if present)
                    if has_bias:
                        imb = ax_list[1].imshow(masked_bias, origin="lower", cmap="gray", vmin=bmin, vmax=bmax)
                        ax_list[1].set_title(f"Used bias frame (bias idx {bias_idx})")
                        fig.colorbar(imb, ax=ax_list[1])

                        # Master flat becomes panel 3 and reduced panel 4
                        im1 = ax_list[2].imshow(masked_flat, origin="lower", cmap="viridis", vmin=fmin, vmax=fmax)
                        ax_list[2].set_title(f"Master flat used (flat idx {flat_idx})")
                        fig.colorbar(im1, ax=ax_list[2])

                        im2 = ax_list[3].imshow(reduced_masked, origin="lower", cmap="gray", vmin=rmin, vmax=rmax)
                        ax_list[3].set_title(f"Reduced science frame (config {key})")
                        fig.colorbar(im2, ax=ax_list[3])

                    else:
                        # No bias: panels are science, flat, reduced
                        im1 = ax_list[1].imshow(masked_flat, origin="lower", cmap="viridis", vmin=fmin, vmax=fmax)
                        ax_list[1].set_title(f"Master flat used (flat idx {flat_idx})")
                        fig.colorbar(im1, ax=ax_list[1])

                        im2 = ax_list[2].imshow(reduced_masked, origin="lower", cmap="gray", vmin=rmin, vmax=rmax)
                        ax_list[2].set_title(f"Reduced science frame (config {key})")
                        fig.colorbar(im2, ax=ax_list[2])

                    plt.tight_layout()
                    plt.show()

                # store reduced image into per-object, per-filter datacube
                try:
                    obj_name = self.instrument.get_header_value(hdul, self.instrument.object_keyword) or "UNKNOWN"
                except Exception:
                    obj_name = "UNKNOWN"
                obj_key = str(obj_name)

                # derive a filter-configuration key from the current setup configuration
                filters = configuration.get("filter") or []
                if isinstance(filters, (list, tuple)):
                    filter_tuple = tuple(filters)
                else:
                    filter_tuple = (filters,)
                # make a compact, filesystem/header-safe key
                try:
                    filter_key = ",".join([str(f) for f in filter_tuple]) if filter_tuple else "UNKNOWN"
                except Exception:
                    filter_key = str(filter_tuple)

                # convert masked array to regular float array with NaNs for masked pixels
                try:
                    arr = reduced_masked.filled(np.nan) if hasattr(reduced_masked, "filled") else np.asarray(reduced_masked)
                    arr = np.asarray(arr, dtype=float)
                except Exception:
                    arr = np.asarray(reduced_masked, dtype=float)

                # ensure top-level object entry exists and is a dict
                if obj_key not in self.reduced_objects or self.reduced_objects.get(obj_key) is None:
                    self.reduced_objects[obj_key] = {}

                obj_dict = self.reduced_objects[obj_key]

                # initialize or append to datacube for this filter configuration
                existing = obj_dict.get(filter_key)
                if existing is None:
                    # start a new datacube with one frame
                    obj_dict[filter_key] = arr[np.newaxis, ...]
                else:
                    # if shapes match, stack; otherwise log and skip this frame
                    if isinstance(existing, np.ndarray) and existing.ndim == 3 and existing.shape[1:] == arr.shape:
                        try:
                            obj_dict[filter_key] = np.concatenate([existing, arr[np.newaxis, ...]], axis=0)
                        except Exception as e:
                            self.logger.warning(f"Failed to append reduced frame for object {obj_key}, filter {filter_key}: {e}")
                    else:
                        self.logger.warning(
                            f"Shape mismatch when adding reduced frame for object {obj_key}, filter {filter_key}: "
                            f"existing={existing.shape if isinstance(existing, np.ndarray) else 'unknown'}, new={arr.shape}; skipping frame"
                        )


        # For every object and filter configuration, compute the median frame and display it
        try:
            for obj_key, obj_dict in getattr(self, "reduced_objects", {}).items():
                for filter_key, cube in (obj_dict or {}).items():
                    try:
                        if not isinstance(cube, np.ndarray):
                            self.logger.warning(f"Skipping non-array datacube for object={obj_key} filter={filter_key}")
                            continue
                        if cube.ndim != 3:
                            self.logger.warning(f"Skipping datacube with invalid ndim for object={obj_key} filter={filter_key}: ndim={cube.ndim}")
                            continue
                        
                        n_frames = int(cube.shape[0])
                        if n_frames < 3:
                            self.logger.info(f"Not enough frames to compute median for object={obj_key} filter={filter_key}: found {n_frames} (need >= 3), skipping")
                            continue

                        sum_img = np.nansum(cube, axis=0)
                        plt.imshow(sum_img, origin="lower", cmap="inferno")
                        plt.title(f"Sum of {n_frames} frames for object={obj_key} filter={filter_key}")
                        plt.show()
                        
                        # compute sigma-clipped median using ccdproc for better outlier rejection
                        # Convert cube to list of CCDData objects for ccdproc
                        ccd_list = []
                        for i in range(cube.shape[0]):
                            frame = cube[i]
                            ccd_data = CCDData(frame, unit=u.adu)
                            ccd_list.append(ccd_data)
                        
                        # Create combiner with sigma clipping
                        combiner = Combiner(ccd_list)
                        combiner.sigma_clipping(low_thresh=0.5, high_thresh=0.5, func=np.ma.median)
                        
                        # Combine using sigma-clipped median
                        median_combined = combiner.median_combine()
                        median_img = median_combined.data

                        # plot the median together with all frames used to create it
                        try:
                            n_frames = int(cube.shape[0])
                            n_panels = n_frames + 1  # median + each frame

                            # limit the number of panels shown to avoid huge figures
                            max_panels = 24
                            if n_panels > max_panels:
                                self.logger.info(f"Too many frames ({n_frames}) to display; showing first {max_panels-1} frames plus median")
                                n_display = max_panels
                            else:
                                n_display = n_panels

                            ncols = min(6, n_display)
                            nrows = int(math.ceil(n_display / ncols))

                            fig, axs = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows))
                            axs_flat = axs.flatten() if hasattr(axs, "flatten") else np.array([axs])

                            # compute a common display stretch from the data used (5th-95th percentiles)
                            all_vals = cube.reshape(-1)
                            finite_vals = all_vals[np.isfinite(all_vals)]
                            if finite_vals.size == 0:
                                vmin, vmax = 0.0, 1.0
                            else:
                                vmin = float(np.nanpercentile(finite_vals, 5))
                                vmax = float(np.nanpercentile(finite_vals, 95))
                                if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
                                    vmin = float(np.nanmin(finite_vals))
                                    vmax = float(np.nanmax(finite_vals)) if np.isfinite(np.nanmax(finite_vals)) else vmin + 1.0

                            # Plot median in the first panel
                            ax0 = axs_flat[0]
                            im = ax0.imshow(median_img, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
                            ax0.set_title(f"Median ({n_frames} frames)")
                            ax0.axis("off")

                            # Plot up to (n_display-1) frames (median occupies index 0)
                            n_frames_to_show = n_display - 1
                            for i in range(n_frames_to_show):
                                ax = axs_flat[i + 1]
                                frame = cube[i]
                                ax.imshow(frame, origin="lower", cmap="gray", vmin=vmin, vmax=vmax)
                                ax.set_title(f"Frame {i}")
                                ax.axis("off")

                            # Turn off any unused subplots
                            for j in range(n_display, axs_flat.size):
                                try:
                                    axs_flat[j].axis("off")
                                except Exception:
                                    pass

                            # shared colorbar for the figure
                            try:
                                fig.colorbar(im, ax=axs_flat.tolist(), fraction=0.02, pad=0.01)
                            except Exception:
                                pass

                            plt.tight_layout()
                            plt.show()
                        except Exception as e:
                            self.logger.warning(f"Failed to plot median + frames for object={obj_key} filter={filter_key}: {e}")

                        # subtract the median from every frame in the datacube and display the residuals
                        diffs = cube - median_img[np.newaxis, ...]  # broadcast median to cube shape

                        for i in range(diffs.shape[0]):
                            frame = diffs[i]
                            orig = cube[i]

                            # robust symmetric stretch for the diff panel
                            try:
                                vmax = np.nanpercentile(np.abs(frame), 95)
                                if not np.isfinite(vmax) or vmax == 0:
                                    vmax = float(np.nanmax(np.abs(frame))) if np.isfinite(np.nanmax(np.abs(frame))) else 1.0
                            except Exception:
                                vmax = 1.0

                            # robust limits for the median image (use 5th-95th percentiles)
                            try:
                                mmin = np.nanpercentile(median_img, 5)
                                mmax = np.nanpercentile(median_img, 95)
                                if not np.isfinite(mmin) or not np.isfinite(mmax) or mmin == mmax:
                                    mmin = float(np.nanmin(median_img)) if np.isfinite(np.nanmin(median_img)) else 0.0
                                    mmax = float(np.nanmax(median_img)) if np.isfinite(np.nanmax(median_img)) else mmin + 1.0
                            except Exception:
                                mmin = float(np.nanmin(median_img)) if np.isfinite(np.nanmin(median_img)) else 0.0
                                mmax = float(np.nanmax(median_img)) if np.isfinite(np.nanmax(median_img)) else mmin + 1.0

                            # robust limits for the original frame (5th-95th percentiles)
                            try:
                                omin = np.nanpercentile(orig, 5)
                                omax = np.nanpercentile(orig, 95)
                                if not np.isfinite(omin) or not np.isfinite(omax) or omin == omax:
                                    omin = float(np.nanmin(orig)) if np.isfinite(np.nanmin(orig)) else 0.0
                                    omax = float(np.nanmax(orig)) if np.isfinite(np.nanmax(orig)) else omin + 1.0
                            except Exception:
                                omin = float(np.nanmin(orig)) if np.isfinite(np.nanmin(orig)) else 0.0
                                omax = float(np.nanmax(orig)) if np.isfinite(np.nanmax(orig)) else omin + 1.0

                            # plot original, median and diff side-by-side
                            fig, axs = plt.subplots(1, 3, figsize=(18, 5))

                            im0 = axs[0].imshow(orig, origin="lower", cmap="gray", vmin=omin, vmax=omax)
                            axs[0].set_title(f"Original frame (Object={obj_key} Filter={filter_key})")
                            axs[0].axis("off")
                            cbar0 = fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
                            cbar0.ax.set_ylabel("ADU")

                            im1 = axs[1].imshow(median_img, origin="lower", cmap="viridis", vmin=mmin, vmax=mmax)
                            axs[1].set_title("Median image")
                            axs[1].axis("off")
                            cbar1 = fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)
                            cbar1.ax.set_ylabel("ADU")

                            im2 = axs[2].imshow(frame, origin="lower", cmap="gray", vmin=-vmax, vmax=vmax)
                            axs[2].set_title(f"Frame - median (frame {i})")
                            axs[2].axis("off")
                            cbar2 = fig.colorbar(im2, ax=axs[2], fraction=0.046, pad=0.04)
                            cbar2.ax.set_ylabel("ADU")

                            plt.tight_layout()
                            plt.show()

                            # plot a 1D vertical slice down the middle of the median-subtracted image
                            try:
                                ny, nx = frame.shape
                                mid_col = nx // 2
                                rows = np.arange(ny)

                                diff_col = frame[:, mid_col]
                                orig_col = orig[:, mid_col]
                                med_col = median_img[:, mid_col]

                                plt.figure(figsize=(8, 4))
                                plt.plot(rows, diff_col, marker='.', linestyle='-', label='Diff (frame - median)', color='C0')
                                plt.plot(rows, orig_col, marker='.', linestyle='--', label='Original', color='C1', alpha=0.7)
                                plt.plot(rows, med_col, marker='.', linestyle=':', label='Median', color='C2', alpha=0.7)
                                plt.xlabel('Row (pixel)')
                                plt.ylabel('ADU')
                                plt.title(f'Central column slice (col={mid_col}) Object={obj_key} Filter={filter_key} Frame={i}')
                                plt.legend(loc='best')
                                plt.grid(alpha=0.3, linestyle=':')
                                plt.tight_layout()
                                plt.show()
                            except Exception as e:
                                self.logger.warning(f"Could not plot central column slice for object={obj_key} filter={filter_key} frame={i}: {e}")

                    
                    except Exception as e:
                        self.logger.error(f"Failed to display median for object={obj_key} filter={filter_key}: {e}")
        except Exception as e:
            self.logger.error(f"Error while iterating reduced objects to show medians: {e}")
        


    def run_pipeline(self):

        if True:

            self.sort_data()

            self.create_setup_table()

            self.determine_bias_configurations()

            self.make_master_bias()

            self.determine_flat_configurations()

            self.make_master_flats()

            self.reduce()


        if False:

            self.load_bias_configurations()

            self.load_setup_table()

            self.load_science_to_bias_map()

            self.load_master_biases()

            self.load_bad_pixel_masks_bias()

            self.load_flat_configurations()

            self.load_science_to_flat_map()

            self.load_bad_pixel_masks_flats()

            self.load_master_flats()

            self.reduce()

        #self.reduce()
