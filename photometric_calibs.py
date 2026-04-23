import numpy as np
from astropy.io import fits
import os
import shlex
import subprocess
import matplotlib.pyplot as plt
import json
from IO import open_fits_file
from matplotlib.patches import Ellipse
from astropy.table import Table
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import astropy.units as u

from astroquery.sdss import SDSS
from astroquery.vizier import Vizier
from astroquery.vizier import Vizier as _Vizier


class Photometric_parser:

    def __init__(self, reduced_dir, logger, object_setup, show_plots=False):

        self.reduced_dir = reduced_dir
        self.logger = logger
        self.object_setup = object_setup
        self.show_plots = show_plots

    def run(self):
        # dummy for inheritance
        pass

    def determine_configurations(self):
        # TODO might not need this - keep for API
        pass

    # these are identical, maybe refactor or just leave out at all
    def run_sex(self, command):
        print("Executing:", " ".join(shlex.quote(a) for a in command))
        subprocess.run(command, cwd=self.reduced_dir, check=True)

    def run_scamp(self, command):
        print("Executing:", " ".join(shlex.quote(a) for a in command))
        subprocess.run(command, cwd=self.reduced_dir, check=True)

    def run_swarp(self, command):
        print("Executing:", " ".join(shlex.quote(a) for a in command))
        subprocess.run(command, cwd=self.reduced_dir, check=True)

    def plot_apertures(self, masked_frame, good_objects, filename):
        """
        Helper method to plot SExtractor apertures to avoid repeated code
        """

        fig, ax = plt.subplots()
        vmin = np.percentile(masked_frame.compressed(), 5)
        vmax = np.percentile(masked_frame.compressed(), 95)
        im = ax.imshow(masked_frame, cmap="gray", origin="lower", vmin=vmin, vmax=vmax)
        plt.colorbar(im, ax=ax)

        # Determine which radius field is available and build masks accordingly
        names = good_objects.names
        has_kron = "KRON_RADIUS" in names
        has_fluxrad = "FLUX_RADIUS" in names

        # ensure valid numeric entries for positions and at least one radius field
        mask_valid = np.isfinite(good_objects["XWIN_IMAGE"]) & np.isfinite(
            good_objects["YWIN_IMAGE"]
        )
        if has_kron:
            mask_valid &= np.isfinite(good_objects["KRON_RADIUS"])
        elif has_fluxrad:
            mask_valid &= np.isfinite(good_objects["FLUX_RADIUS"])

        xs = good_objects["XWIN_IMAGE"][mask_valid]
        ys = good_objects["YWIN_IMAGE"][mask_valid]

        # parse Kron factor from the SExtractor config (PHOT_AUTOPARAMS)
        kron_factor = 2.5  # default fallback
        try:
            with open(self.sex_config, "r") as f:
                for line in f:
                    if "PHOT_AUTOPARAMS" in line and not line.strip().startswith("#"):
                        rhs = line.split("PHOT_AUTOPARAMS", 1)[1]
                        rhs = rhs.split("#")[0]
                        rhs = rhs.replace(",", " ")
                        parts = rhs.split()
                        for p in parts:
                            try:
                                kron_factor = float(p)
                                break
                            except ValueError:
                                continue
                        break
        except Exception:
            pass

        if has_kron:
            kron_vals = good_objects["KRON_RADIUS"][mask_valid].astype(float)
            # If we have A_IMAGE/B_IMAGE/THETA_IMAGE, draw ellipses scaled by kron
            if "A_IMAGE" in names and "B_IMAGE" in names:
                a_vals = good_objects["A_IMAGE"][mask_valid].astype(float)
                b_vals = good_objects["B_IMAGE"][mask_valid].astype(float)
                theta_vals = (
                    good_objects["THETA_IMAGE"][mask_valid].astype(float)
                    if "THETA_IMAGE" in names
                    else np.zeros_like(a_vals)
                )
                for x, y, k, a, b, theta in zip(
                    xs, ys, kron_vals, a_vals, b_vals, theta_vals
                ):
                    # A_IMAGE and B_IMAGE are semi-axis (pixels). Width/height for Ellipse are total extents.
                    width = 2.0 * k * kron_factor * a
                    height = 2.0 * k * kron_factor * b
                    ell = Ellipse(
                        (x, y),
                        width=width,
                        height=height,
                        angle=theta,
                        edgecolor="red",
                        facecolor="none",
                        linewidth=1,
                    )
                    ax.add_patch(ell)
            else:
                # fallback: draw circular apertures with radius = kron * kron_factor
                for x, y, k in zip(xs, ys, kron_vals):
                    circ = plt.Circle(
                        (x, y),
                        k * kron_factor,
                        edgecolor="red",
                        facecolor="none",
                        linewidth=1,
                    )
                    ax.add_patch(circ)
        elif has_fluxrad:
            rs = good_objects["FLUX_RADIUS"][mask_valid].astype(float)
            for x, y, r in zip(xs, ys, rs):
                circ = plt.Circle(
                    (x, y), r, edgecolor="red", facecolor="none", linewidth=1
                )
                ax.add_patch(circ)
        else:
            self.logger.warning(
                "No radius parameter available to plot apertures (KRON_RADIUS or FLUX_RADIUS)."
            )
        ax.set_title("Detected Objects with FLAGS=0")
        ax.set_xlabel("XWIN_IMAGE")
        ax.set_ylabel("YWIN_IMAGE")

        save_path = os.path.join(
            self.reduced_dir, "apertures_" + filename.replace(".fits", ".png")
        )
        plt.savefig(save_path)
        if self.show_plots:
            plt.show()
        else:
            plt.close()

    def query_SDSS(self, catalog_path, frame_path, filter_string, filter_error_string, object_name="unknown_object"):
        """
        Returns:
        0 on success, -1 on failure (e.g. no valid matches found)
        """

        with fits.open(os.path.join(self.reduced_dir, catalog_path)) as hdul:
            data_table = Table(hdul[2].data)
        with fits.open(os.path.join(self.reduced_dir, frame_path)) as hdul:
            hdu = hdul[0]
            wcs = WCS(hdu.header)

        ra, dec = wcs.all_pix2world(
            data_table["XWIN_IMAGE"], data_table["YWIN_IMAGE"], 0
        )

        source_coords = SkyCoord(ra * u.deg, dec * u.deg)

        center = SkyCoord(np.mean(ra) * u.deg, np.mean(dec) * u.deg)



        radius = u.Quantity(3, u.arcmin)  # adjust to FoV

        # TODO: sensible matching for different filters
        sdss = SDSS.query_region(
            center,
            radius=radius,
            photoobj_fields=["ra", "dec", filter_string, filter_error_string],
        )

        if sdss is None or len(sdss) == 0:
            self.logger.error("No SDSS sources found in the field for photometric calibration.")
            return -1


        sdss_coords = SkyCoord(sdss["ra"] * u.deg, sdss["dec"] * u.deg)

        idx, sep2d, _ = source_coords.match_to_catalog_sky(sdss_coords)

        max_sep = 1.0 * u.arcsec

        good_sep = sep2d < max_sep
        # -------------------------------
        # FILTER SDSS QUALITY FIRST
        # -------------------------------

        good_sdss = (sdss[filter_string] > 0) & (  # removes -9999 / junk
            sdss[filter_error_string] < 0.2
        )  # reasonable photometry

        # -------------------------------
        # FILTER SExtractor flux
        # -------------------------------

        good_inst = np.isfinite(data_table["MAG_AUTO"])

        # -------------------------------
        # COMBINE ALL CONDITIONS
        # -------------------------------

        good = good_sep & good_sdss[idx] & good_inst

        matched_sources = data_table[good]

        matched_sdss = sdss[idx[good]]

        instrumental_mag = matched_sources["MAG_AUTO"]

        sdss_mag = matched_sdss[filter_string]

        zp_values = sdss_mag - instrumental_mag

        # extra safety: remove any remaining bad values

        finite = np.isfinite(zp_values)

        zp_values = zp_values[finite]

        if len(zp_values) == 0:
            self.logger.error("No valid zeropoint values found after filtering.")
            return

        self.logger.info(f"Zeropoint values: {zp_values}")

        zp = np.median(zp_values)

        self.logger.info(f"Median Zeropoint: {zp}")

        plt.hist(zp_values, bins=len(zp_values), color='C0', alpha=0.8)
        plt.axvline(zp, color='C1', linestyle='--', label=f'median = {zp:.3f}')
        plt.xlabel('Zeropoint (mag)')
        plt.ylabel('Number of sources')
        plt.title(f'Zeropoint distribution (median = {zp:.3f})')
        plt.legend()
        plt.grid(True, ls=':', alpha=0.6)
        plt.tight_layout()
        if self.show_plots:
            plt.show()
        else:
            plt.close()
        
        # write the zeropoints to the table
        data_table[f"ZP_{filter_string}"] = np.nan

        # basic quality mask (instrumental sanity checks)
        good = (
            np.isfinite(data_table["MAG_AUTO"]) &
            np.isfinite(data_table["FLUX_AUTO"]) &
            (data_table["FLUX_AUTO"] > 0)
        )

        # apply zeropoint to create calibrated magnitude array
        calibrated_mag = np.full(len(data_table), np.nan)
        calibrated_mag[good] = data_table["MAG_AUTO"][good] + zp

        # keep RA/DEC in degrees on the full table for compatibility
        data_table["RA"] = ra
        data_table["DEC"] = dec

        # build a compact table containing only sexagesimal RA (hh:mm:ss), DEC (dd:mm:ss), and calibrated magnitude
        coords = SkyCoord(ra * u.deg, dec * u.deg)
        ra_str = coords.ra.to_string(unit=u.hour, sep=":", pad=True, precision=2)
        dec_str = coords.dec.to_string(unit=u.deg, sep=":", alwayssign=True, pad=True, precision=2)

        valid = np.isfinite(calibrated_mag)

        out_table = Table(
            [ra_str[valid], dec_str[valid], calibrated_mag[valid]],
            names=["RA", "DEC", f"MAG_CAL_{filter_string}"],
        )

        out_path = os.path.join(self.reduced_dir, f"calibrated_SDSS_{object_name}_{filter_string}.fits")
        out_table.write(out_path, overwrite=True)

        return 0



 
        
    def query_PanSTARRS(self, catalog_path, frame_path, filter_string, filter_error_string, object_name="unknown_object"):
        """
        Query Pan-STARRS via MAST Catalogs for PS1 photometry and compute zeropoint.
        Expects filter_string like 'gMeanPSFMag' and filter_error_string like 'gMeanPSFMagErr'
        Returns 0 on success, -1 on failure.
        """
        from astroquery.mast import Catalogs

        # load SExtractor table and frame WCS
        try:
            with fits.open(os.path.join(self.reduced_dir, catalog_path)) as hdul:
                data_table = Table(hdul[2].data)
            with fits.open(os.path.join(self.reduced_dir, frame_path)) as hdul:
                hdu = hdul[0]
                wcs = WCS(hdu.header)
        except Exception as e:
            self.logger.error(f"Failed to open catalog/frame: {e}")
            return -1

        # pixel -> world
        ra, dec = wcs.all_pix2world(data_table["XWIN_IMAGE"], data_table["YWIN_IMAGE"], 0)
        source_coords = SkyCoord(ra * u.deg, dec * u.deg)
        center = SkyCoord(np.mean(ra) * u.deg, np.mean(dec) * u.deg)
        radius = 3 * u.arcmin

        # build columns to request from MAST PanSTARRS
        mag_col = filter_string
        err_col = filter_error_string
        cols = ["raMean", "decMean", mag_col, err_col]

        try:
            res = Catalogs.query_region(center, radius=radius, catalog="Panstarrs", columns=cols)
        except Exception as e:
            self.logger.error(f"Pan-STARRS (MAST) query failed: {e}")
            return -1

        if res is None or len(res) == 0:
            self.logger.error("No Pan-STARRS sources found in the field (MAST).")
            return -1

        # ensure columns exist, handle small variations
        if "raMean" not in res.colnames or "decMean" not in res.colnames:
            # try fallback names
            if "ra" in res.colnames and "dec" in res.colnames:
                ra_key, dec_key = "ra", "dec"
            else:
                self.logger.error(f"Pan-STARRS result missing RA/DEC columns: {res.colnames}")
                return -1
        else:
            ra_key, dec_key = "raMean", "decMean"

        if mag_col not in res.colnames or err_col not in res.colnames:
            self.logger.error(f"Pan-STARRS result missing requested mag columns: {mag_col}, {err_col}")
            return -1

        ps_ra = np.asarray(res[ra_key]).astype(float)
        ps_dec = np.asarray(res[dec_key]).astype(float)
        ps_mag = np.asarray(res[mag_col])
        ps_err = np.asarray(res[err_col])

        pan_coords = SkyCoord(ps_ra * u.deg, ps_dec * u.deg)

        # match sources
        idx, sep2d, _ = source_coords.match_to_catalog_sky(pan_coords)
        max_sep = 1.0 * u.arcsec
        good_sep = sep2d < max_sep

        # quality filters for Pan-STARRS: valid mag and reasonable error
        good_ps = np.isfinite(ps_mag) & (ps_mag > -90) & np.isfinite(ps_err) & (ps_err < 0.5)

        # SExtractor instrument sanity
        good_inst = np.isfinite(data_table["MAG_AUTO"])

        # combine masks (indexing into Pan-STARRS catalog via idx for each source)
        good = good_sep & good_inst & good_ps[idx]

        matched_sources = data_table[good]
        if len(matched_sources) == 0:
            self.logger.error("No matched sources after applying quality cuts for Pan-STARRS (MAST).")
            return -1

        matched_panstarrs = res[idx[good]]

        instrumental_mag = matched_sources["MAG_AUTO"]
        pan_mag = np.asarray(matched_panstarrs[mag_col])

        zp_values = pan_mag - instrumental_mag
        zp_values = zp_values[np.isfinite(zp_values)]

        if len(zp_values) == 0:
            self.logger.error("No valid zeropoint values found after filtering (Pan-STARRS).")
            return -1

        zp = np.median(zp_values)
        self.logger.info(f"Pan-STARRS Zeropoint values: {zp_values}")
        self.logger.info(f"Pan-STARRS Median Zeropoint: {zp:.3f}")

        # plot histogram for QA
        plt.hist(zp_values, bins=max(6, min(50, len(zp_values))), color="C0", alpha=0.8)
        plt.axvline(zp, color="C1", linestyle="--", label=f"median = {zp:.3f}")
        plt.xlabel("Zeropoint (mag)")
        plt.ylabel("Number of sources")
        plt.title(f"Pan-STARRS Zeropoint distribution (median = {zp:.3f})")
        plt.legend()
        plt.grid(True, ls=":", alpha=0.6)
        plt.tight_layout()
        if self.show_plots:
            plt.show()
        else:
            plt.close()

        # write zeropoint to the main table and produce calibrated magnitudes
        data_table[f"ZP_{mag_col}"] = np.nan

        # astropy Table doesn't provide .get — handle missing FLUX_AUTO explicitly
        if "FLUX_AUTO" in data_table.colnames:
            flux_vals = np.asarray(data_table["FLUX_AUTO"], dtype=float)
        else:
            flux_vals = np.full(len(data_table), np.nan)

        good_table = (
            np.isfinite(data_table["MAG_AUTO"])
            & np.isfinite(flux_vals)
            & (flux_vals > 0)
        )
        calibrated_mag = np.full(len(data_table), np.nan)
        calibrated_mag[good_table] = data_table["MAG_AUTO"][good_table] + zp

        # attach RA/DEC in degrees
        data_table["RA"] = ra
        data_table["DEC"] = dec

        coords = SkyCoord(ra * u.deg, dec * u.deg)
        ra_str = coords.ra.to_string(unit=u.hour, sep=":", pad=True, precision=2)
        dec_str = coords.dec.to_string(unit=u.deg, sep=":", alwayssign=True, pad=True, precision=2)
        valid = np.isfinite(calibrated_mag)

        out_table = Table(
            [ra_str[valid], dec_str[valid], calibrated_mag[valid]],
            names=["RA", "DEC", f"MAG_CAL_{mag_col}"],
        )
        out_path = os.path.join(self.reduced_dir, f"calibrated_{object_name}_panstarrs_{filter_string}.fits")
        out_table.write(out_path, overwrite=True)

        return 0
        

    def query_WISE():
        # TODO implement this to query WISE for photometric calibration
        pass

    def query_2MASS():
        # TODO implement this to query 2MASS for photometric calibration
        pass

    def query_GAIA():
        # TODO implement this to query GAIA for photometric calibration
        pass

    def query_HST():
        # TODO implement this to query HST for photometric calibration
        pass


class ALFOSC_parser(Photometric_parser):

    def __init__(self, reduced_dir, logger, object_setup, show_plots=False):

        # TODO make a global directory for these
        self.sex_config = os.path.normpath(os.path.join(reduced_dir, "default.sex"))
        self.sex_param = os.path.normpath(os.path.join(reduced_dir, "run1.param"))

        # TODO expland with more filters as needed
        self.filter_query_mapping_SDSS = {
            "u'": ["psfMag_u", "psfMagErr_u"],
            "g'": ["psfMag_g", "psfMagErr_g"],
            "r'": ["psfMag_r", "psfMagErr_r"],
            "i'": ["psfMag_i", "psfMagErr_i"],
            "z'": ["psfMag_z", "psfMagErr_z"],
        }

        self.filter_query_mapping_PanSTARRS = {
            "g'": ["gMeanPSFMag", "gMeanPSFMagErr"],
            "r'": ["rMeanPSFMag", "rMeanPSFMagErr"],
            "i'": ["iMeanPSFMag", "iMeanPSFMagErr"],
            "z'": ["zMeanPSFMag", "zMeanPSFMagErr"],
        }

        self.mask_x_fraction = 0.1
        self.mask_y_fraction = 0.1

        super().__init__(reduced_dir, logger, object_setup, show_plots=show_plots)

    def run(self):

        for obj_key, object_info in self.object_setup.items():

            for object_name in object_info.keys():

                info = object_info[object_name]

                filters = info.get("filter", [])

                # remove all entries that are not "Open"
                filters = [f for f in filters if not f.startswith("Open")]

                # ensure only one filter remains
                if len(filters) == 0:
                    self.logger.error(
                        f"No valid filters found for {object_name} in {obj_key}."
                    )
                    continue

                if len(filters) > 1:
                    self.logger.error(
                        f"Multiple filters found for {object_name} in {obj_key}, only single filters are supported. Skipping..."
                    )
                    continue

                self.logger.info(
                    f"Processing {object_name} in {obj_key} with filter {filters[0]}"
                )

                stripped_filter_name = filters[0].split("_", 1)[0].strip()

                if stripped_filter_name not in self.filter_query_mapping_SDSS.keys():
                    self.logger.error(
                        f"Filter {filters[0]} not implemented. Skipping..."
                    )
                    continue

                files = info.get("files", [])

                # weights will be created from BPM's
                weights_filenames = []

                if files is None or len(files) == 0:
                    self.logger.error(
                        f"No files found for {object_name} in {obj_key}. Skipping..."
                    )
                    continue

                stack = False

                if len(files) == 1:
                    self.logger.info(
                        "Only single exposure found, no stacking is needed"
                    )

                else:
                    self.logger.info(
                        f"Multiple exposures found for {object_name} in {obj_key}, stacking will be performed"
                    )
                    stack = True

                # first step - SExtractor on all frames

                for file in files:

                    file_path = os.path.join(
                        self.reduced_dir, "reduced_science_" + file
                    )

                    hdul = open_fits_file(file_path, self.logger)

                    data = hdul[1].data
                    bpm = hdul[2].data

                    y_size, x_size = bpm.shape
                    x_start = int(x_size * self.mask_x_fraction)
                    y_start = int(y_size * self.mask_y_fraction)
                    x_end = int(x_size * (1 - self.mask_x_fraction))
                    y_end = int(y_size * (1 - self.mask_y_fraction))

                    # we use this as a weight map for source extraction
                    inverted_bpm = np.where(bpm == 0, 1, 0)
                    inverted_bpm[0:y_start, :] = 0
                    inverted_bpm[y_end:, :] = 0
                    inverted_bpm[:, 0:x_start] = 0
                    inverted_bpm[x_end, :] = 0

                    # write to disc to comply so it can be used by SExtractor

                    weights_filename = os.path.join(
                        self.reduced_dir, file.replace(".fits", "_weights.fits")
                    )

                    weights_filenames.append(weights_filename)

                    fits.writeto(
                        weights_filename, inverted_bpm.astype(np.uint8), overwrite=True
                    )

                    cat_filename = os.path.join(
                        self.reduced_dir,
                        "reduced_science_" + file.replace(".fits", ".cat"),
                    )

                    cmd = [
                        "sex",
                        "-c",
                        self.sex_config,
                        file_path + "[1]",
                        "-PARAMETERS_NAME",
                        self.sex_param,
                        "-WEIGHT_TYPE",
                        "MAP_WEIGHT",
                        "-WEIGHT_IMAGE",
                        weights_filename,
                        "-CATALOG_NAME",
                        cat_filename,
                        "-VERBOSE_TYPE",
                        "FULL",
                    ]

                    self.run_sex(cmd)

                    with fits.open(cat_filename) as hdul_table:
                        data_table = hdul_table[2].data

                    good_objects = data_table[data_table["FLAGS"] == 0]

                    masked_frame = np.ma.masked_where(bpm == 1, data)

                    self.plot_apertures(masked_frame, good_objects, file)

                    # now run scamp right away
                    scamp_cmd = [
                        "scamp",
                        cat_filename,
                        "-ASTREF_CATALOG",
                        "GAIA-EDR3",
                        "-ASTREF_BAND",
                        "G",
                        "-VERBOSE_TYPE",
                        "NORMAL",
                        "-CHECKPLOT_TYPE",
                        "NONE",
                    ]

                    self.run_scamp(scamp_cmd)

                final_result_name = f"{object_name}.fits"
                final_weights_filename = f"{object_name}_weights.fits"

                if stack:
                    # prepare the lists for stacking using SWarp

                    file_list_name = os.path.join(
                        self.reduced_dir, f"{object_name}_files.list"
                    )

                    weights_list_name = os.path.join(
                        self.reduced_dir, f"{object_name}_weights.list"
                    )

                    with open(file_list_name, "w") as f:
                        for file in files:
                            f.write(
                                os.path.join(
                                    self.reduced_dir, "reduced_science_" + file + "[1]"
                                )
                                + "\n"
                            )

                    with open(weights_list_name, "w") as f:
                        for weights_file in weights_filenames:
                            f.write(weights_file + "\n")

                    swarp_cmd = [
                        "SWarp",
                        "@" + file_list_name,
                        "-c",
                        "default.swarp",
                        "-IMAGEOUT_NAME",
                        final_result_name,
                        "-COMBINE_TYPE",
                        "MEDIAN",
                        "-WEIGHT_IMAGE",
                        "@" + weights_list_name,
                        "-WEIGHTOUT_NAME",
                        f"swarp_{object_name}_weights.fits",
                        "-PIXEL_SCALE",
                        "0.2138",
                        "-PIXEL_SCALE_TYPE",
                        "MANUAL",
                        "-CENTER_TYPE",
                        "MOST",
                        "-SUBTRACT_BACK",
                        "N",
                    ]

                    self.run_swarp(swarp_cmd)

                    final_result_path = os.path.join(
                        self.reduced_dir, final_result_name
                    )
                    with fits.open(final_result_path) as hdul_final:
                        final_data = hdul_final[0].data


                    # we save the header to force the same transform on the bpm

                    hdr = fits.getheader(
                        os.path.join(self.reduced_dir, final_result_name), ext=0
                    )
                    fits.writeto(
                        final_result_name.replace(".fits", ".head"),
                        data=None,
                        header=hdr,
                        overwrite=True,
                    )

                    x, y = hdr["NAXIS1"], hdr["NAXIS2"]

                    # now run it blindly on the weight stack with nearest neibhor to get the combined mask (this is a hack since swarp doesn't support separate bad pixel masks, but it will at least show us which pixels are masked in the final stack)
                    swarp_mask_cmd = [
                        "SWarp",
                        "@" + weights_list_name,
                        "-c",
                        "default.swarp",
                        "-IMAGEOUT_NAME",
                        final_weights_filename,
                        "-IMAGE_SIZE",
                        f"{x},{y}",
                        # lock geometry to science stack
                        "-HEADER_NAME",
                        final_result_name.replace(".fits", ".head"),
                        # mask-safe settings
                        "-RESAMPLING_TYPE",
                        "NEAREST",
                        "-COMBINE_TYPE",
                        "MIN",
                        # disable all weighting logic
                        "-WEIGHT_TYPE",
                        "NONE",
                        "-WEIGHT_IMAGE",
                        "",
                        # disable image “science processing”
                        "-SUBTRACT_BACK",
                        "N",
                        "-RESCALE_WEIGHTS",
                        "N",
                        "-FSCALE_KEYWORD",
                        "NONE",
                        "-FSCALE_DEFAULT",
                        "1.0",
                    ]

                    self.run_swarp(swarp_mask_cmd)

                    # now read the weights and convert to a bpm

                    with fits.open(
                        os.path.join(self.reduced_dir, final_weights_filename)
                    ) as hdul_weights:
                        weights_data = hdul_weights[0].data

                    bpm_data = np.where(weights_data == 0, 1, 0)

                    # now load the final result and plot for QA


                    masked_final = np.ma.masked_where(bpm_data == 1, final_data)

                    plt.imshow(
                        masked_final,
                        cmap="gray",
                        origin="lower",
                        vmin=np.percentile(masked_final.compressed(), 5),
                        vmax=np.percentile(masked_final.compressed(), 95),
                    )
                    plt.colorbar()
                    plt.title(
                        f"Final stacked image for {object_name} with masked pixels."
                    )
                    plt.xlabel("X Pixel")
                    plt.ylabel("Y Pixel")
                    save_path = os.path.join(
                        self.reduced_dir, f"final_{object_name}.png"
                    )
                    plt.savefig(save_path)
                    if self.show_plots:
                        plt.show()
                    else:
                        plt.close()

                    # now run the SExtractor on the final stack

                    final_cat_filename = os.path.join(
                        self.reduced_dir, final_result_name.replace(".fits", ".cat")
                    )

                    # TODO later disable bacground sub when using already subtracted images

                    cmd = [
                        "sex",
                        "-c",
                        self.sex_config,
                        final_result_name,
                        "-PARAMETERS_NAME",
                        self.sex_param,
                        "-WEIGHT_TYPE",
                        "MAP_WEIGHT",
                        "-WEIGHT_IMAGE",
                        final_weights_filename,
                        "-CATALOG_NAME",
                        final_cat_filename,
                        "-VERBOSE_TYPE",
                        "FULL",
                    ]

                    self.run_sex(cmd)

                    with fits.open(final_cat_filename) as hdul_table:
                        data_table = hdul_table[2].data

                    good_objects = data_table[data_table["FLAGS"] == 0]

                    self.plot_apertures(masked_final, good_objects, file)

                    query_result = self.query_SDSS(
                        final_cat_filename,
                        final_result_name,
                        self.filter_query_mapping_SDSS[stripped_filter_name][0],
                        self.filter_query_mapping_SDSS[stripped_filter_name][1],
                        object_name=object_name,
                    )
                    
                    if query_result == 0:
                        self.logger.info(f"Photometric calibration successful for {object_name} in {obj_key}.")
                    else:
                        self.logger.error(f"Photometric calibration failed for {object_name} in {obj_key} using SDSS.")

                        self.logger.info(f"Attempting photometric calibration for {object_name} in {obj_key} using PanSTARRS.")

                        if self.filter_query_mapping_PanSTARRS.get(stripped_filter_name) is not None:
                            
                            self.logger.info(f"Attempting photometric calibration for {object_name} in {obj_key} using PanSTARRS.")

                            self.query_PanSTARRS(final_cat_filename, final_result_name, self.filter_query_mapping_PanSTARRS[stripped_filter_name][0], self.filter_query_mapping_PanSTARRS[stripped_filter_name][1], object_name=object_name) 

class NOTCAM_parser(Photometric_parser):

    def __init__(self, reduced_dir, logger, object_setup, show_plots=False):
        super().__init__(reduced_dir, logger, object_setup, show_plots=show_plots)

    def run(self):

        print(self.object_setup)
