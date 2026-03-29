from dataclasses import dataclass
from typing import Optional
import os
from enum import Enum
from datatypes import ImageType
from ccdproc import combine, CCDData, subtract_dark, subtract_bias, flat_correct
from IO import open_fits_file, write_frame, read_frame, get_header_value
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.io import fits
from datetime import datetime
import re


@dataclass
class Detector:
    gain: Optional[tuple[str, int]] = None
    read_noise: Optional[tuple[str, int]] = None
    saturation_level: Optional[tuple[str, int]] = None
    dark_current: Optional[tuple[str, int]] = None
    pixel_scale: Optional[tuple[str, int]] = None
    field_of_view: Optional[tuple[str, int]] = None
    window_keyword: Optional[tuple[str, int]] = None
    bin_x_keyword: Optional[tuple[str, int]] = None
    bin_y_keyword: Optional[tuple[str, int]] = None
    bpm_median_threshold: Optional[float] = 0.2


@dataclass
class Telescope:
    name: Optional[tuple[str, int]] = None
    aperture: Optional[tuple[str, int]] = None
    focal_length: Optional[tuple[str, int]] = None
    location: Optional[tuple[str, int]] = None


class Instrument:
    def __init__(
        self,
        name: Optional[str] = None,
        detector: Optional[Detector] = None,
        telescope: Optional[Telescope] = None,
        filter_keyword: Optional[tuple[list[str], int]] = None,
        obsmode_keyword: Optional[tuple[str, int]] = None,
        imaging_obsmode_keyword: Optional[tuple[str, int]] = None,
        imagetype_keyword: Optional[tuple[str, int]] = None,
        bias_keyword: Optional[list[str]] = None,
        dark_keyword: Optional[list[str]] = None,
        flat_keyword: Optional[list[str]] = None,
        science_keyword: Optional[list[str]] = None,
        data_hdu_extension: Optional[int] = None,
        object_keyword: Optional[tuple[str, int]] = None,
        exposure_time_keyword: Optional[tuple[str, int]] = None,
    ):
        self.name = name
        self.detector = detector if detector is not None else Detector()
        self.telescope = telescope if telescope is not None else Telescope()

        self.filter_keyword = filter_keyword if filter_keyword is not None else ([], 0)

        self.obsmode_keyword = (
            obsmode_keyword if obsmode_keyword is not None else (None, 0)
        )
        self.imaging_obsmode_keyword = (
            imaging_obsmode_keyword
            if imaging_obsmode_keyword is not None
            else (None, 0)
        )
        self.imagetype_keyword = (
            imagetype_keyword if imagetype_keyword is not None else (None, 0)
        )
        self.bias_keyword = bias_keyword if bias_keyword is not None else []
        self.dark_keyword = dark_keyword if dark_keyword is not None else []
        self.flat_keyword = flat_keyword if flat_keyword is not None else []
        self.science_keyword = science_keyword if science_keyword is not None else []

        self.object_keyword = (
            object_keyword if object_keyword is not None else (None, 0)
        )

        self.exposure_time_keyword = (
            exposure_time_keyword if exposure_time_keyword is not None else (None, 0)
        )

        self.data_hdu_extension = data_hdu_extension

    def match_image_type(self, hdul) -> Optional[ImageType]:

        # Base implementation: intended to be overridden by subclasses.
        # Return None to indicate "no determination" at this level.
        return None

    def update_bad_pixel_map(
        self,
        master_frame,
        logger,
        output_dir,
        key,
        bad_pixel_mask=None,
        show_plots=False,
    ):

        n_median_deviation = self.detector.bpm_median_threshold or 0.2

        # create a new bad pixel mask based on sigma clipping the master frame
        if bad_pixel_mask is None:
            bad_pixel_mask = np.zeros(master_frame.shape, dtype=bool)

        master_frame_copy = master_frame.copy()
        master_frame_masked = np.ma.masked_array(master_frame_copy, mask=bad_pixel_mask)

        median = np.nanmedian(master_frame_masked)

        zero_pixels = master_frame_masked == 0
        logger.info(
            f"Identified {np.sum(zero_pixels)} zero-value pixels in master frame"
        )

        logger.info(
            f"Masking pixels that deviate from median by more than {n_median_deviation} medians (median={median:.2f})"
        )
        new_bad_pixels = np.abs(master_frame_masked - median) > (
            n_median_deviation * median
        )

        all_bad_pixels = zero_pixels | new_bad_pixels

        # combine with existing bad pixel mask

        combined_bad_pixel_mask = bad_pixel_mask | all_bad_pixels

        logger.info(f"Identified {np.sum(all_bad_pixels)} new bad pixels.")

        masked_array = np.ma.masked_array(
            master_frame_masked, mask=combined_bad_pixel_mask
        )

        # plot histogram of pixel values, excluding NaNs
        data_for_hist = masked_array.flatten()
        data_for_hist = data_for_hist[~np.isnan(data_for_hist)]
        n_bins = int(np.sqrt(data_for_hist.size))
        plt.figure()
        plt.hist(data_for_hist, bins=n_bins, color="gray", edgecolor="black")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.title(
            "Pixel Value Distribution - Everything outside the rejection thresholds is masked as bad pixel"
        )

        # vertical lines for median and rejection thresholds
        plt.axvline(
            median,
            color="red",
            linestyle="-",
            linewidth=1.5,
            label=f"Median = {median:.2f}",
        )
        lower = median - n_median_deviation * median
        upper = median + n_median_deviation * median
        plt.axvline(
            lower,
            color="orange",
            linestyle="--",
            linewidth=1.2,
            label=f"Reject < {lower:.2f}",
        )
        plt.axvline(
            upper,
            color="orange",
            linestyle="--",
            linewidth=1.2,
            label=f"Reject > {upper:.2f}",
        )
        plt.legend(loc="upper right")

        info_text = f"median={median:.2f}, rejection: |value - median| > {n_median_deviation}×median (±{n_median_deviation*median:.2f})"
        logger.info(info_text)

        # annotate plot with the same info
        plt.text(
            0.01, 0.95, info_text, transform=plt.gca().transAxes, fontsize=9, va="top"
        )
        plt.grid()

        plt.xlim(
            median - n_median_deviation * median * 1.1,
            median + n_median_deviation * median * 1.1,
        )

        save_path = os.path.join(output_dir, f"bad_pixel_map_{key}.png")
        plt.savefig(save_path)
        logger.info(f"Saved bad pixel map plot to: {save_path}")

        if show_plots:
            plt.show()
        else:
            plt.close()

        return np.asarray(combined_bad_pixel_mask, dtype=bool)

    def make_master_bias(
        self,
        input_dir,
        output_dir,
        bias_setup,
        logger,
        bad_pixel_masks=None,
        show_plots=False,
    ):

        if bad_pixel_masks is None:
            bad_pixel_masks = {}

        master_biases = {}

        for key, value in bias_setup.items():
            logger.info(
                f"Making master bias for setup: {key} (window: {value['window']}, x_bin: {value['bin_x']}, y_bin: {value['bin_y']}) with {len(value['files'])} bias frames"
            )

            bias_stack = []

            for file in value["files"]:

                filepath = os.path.join(input_dir, file)

                hdul = open_fits_file(filepath, logger)

                # create CCDData directly from the numpy array and keep header/meta
                try:
                    data = hdul[self.data_hdu_extension].data
                    hdr = (
                        hdul[self.data_hdu_extension].header
                        if len(hdul) > self.data_hdu_extension
                        else None
                    )
                    ccd_data = CCDData(
                        data,
                        unit=u.adu,
                        meta={"header": hdr} if hdr is not None else None,
                    )
                except Exception:
                    # fallback: create CCDData without header
                    ccd_data = CCDData(hdul[self.data_hdu_extension].data, unit=u.adu)
                bias_stack.append(ccd_data)

            # 2 sigma clipped median master bias
            master_bias = combine(
                bias_stack,
                method="median",
                sigma_clip=True,
                sigma_clip_low_thresh=2,
                sigma_clip_high_thresh=2,
            )

            bad_pixel_mask = self.update_bad_pixel_map(
                master_bias.data,
                logger,
                output_dir,
                key,
                bad_pixel_masks[key] if key in bad_pixel_masks else None,
                show_plots=show_plots,
            )

            if key not in bad_pixel_masks:
                bad_pixel_masks[key] = bad_pixel_mask
            else:
                bad_pixel_masks[key] = bad_pixel_masks[key] | bad_pixel_mask

            masked_bias = np.ma.masked_array(master_bias.data, mask=bad_pixel_mask)

            plt.close()

            plt.imshow(
                masked_bias,
                cmap="gray",
                vmin=np.percentile(master_bias.data, 5),
                vmax=np.percentile(master_bias.data, 95),
                origin="lower",
            )
            plt.colorbar()
            plt.title(
                f"Master Bias for window: {value['window']}, x_bin: {value['bin_x']}, y_bin: {value['bin_y']}"
            )

            # save plot to the output directory
            plot_path = os.path.join(output_dir, f"master_bias_{key}.png")
            plt.savefig(plot_path)

            if show_plots:
                plt.show()

            # open the middle frame, copy its HDUList, replace the data with master_bias and write to disk
            mid_idx = len(value["files"]) // 2
            mid_file = value["files"][mid_idx]
            mid_path = os.path.join(input_dir, mid_file)

            hdul_mid = open_fits_file(mid_path, logger)

            # deep-copy all HDUs so we don't mutate the original HDUList in memory
            hdul_copy = fits.HDUList([hdu.copy() for hdu in hdul_mid])

            write_frame(
                self,
                hdul_copy,
                master_bias.data,
                f"master_bias_{key}.fits",
                output_dir,
                logger,
                bad_pixel_mask=bad_pixel_masks[key],
                comment=f"Master bias created on {datetime.now().isoformat()} using {len(value['files'])} bias frames with 2-sigma clipping. Bad pixel mask updated based on deviation from median and zero-value pixels.",
                header_updates={
                    "MASTERBIAS": (True, "Indicates this frame is a master bias"),
                    "BIASCNT": (len(value["files"]), "Number of bias frames combined"),
                    "BIASWIN": (value["window"], "Window setting for this master bias"),
                    "BIASBINX": (value["bin_x"], "Binning in X for this master bias"),
                    "BIASBINY": (value["bin_y"], "Binning in Y for this master bias"),
                },
            )

            master_biases[key] = master_bias

        logger.info("Master bias creation complete.")

        return master_biases, bad_pixel_masks

    def make_master_flat(
        self,
        input_dir,
        output_dir,
        flat_setup,
        logger,
        bad_pixel_masks=None,
        dark_frames=None,
        bias_frames=None,
        science_to_bias_map=None,
        show_plots=False,
        skip_dark_correction=False,
        skip_bias_correction=False,
    ):
        """
        The bad pixel masks are assumed to be the one used from the dark and
        bias configurations only.
        """

        bad_pixel_masks_science = {}

        master_flats = {}

        # copy these as they might change due to setup-specific logic, but we want to reset for every setup
        skip_dark_correction_input = skip_dark_correction
        skip_bias_correction_input = skip_bias_correction

        for key, value in flat_setup.items():

            n_files = len(value["files"])

            # if no files for this setup, skip
            if n_files == 0:
                logger.warning(
                    f"No flat frames found for setup: {key} (window: {value['window']}, x_bin: {value['bin_x']}, y_bin: {value['bin_y']}, filter: {value['filter']}), skipping master flat creation."
                )
                continue

            # TODO make this a fallback to query then calib database afterwards afterwards
            if n_files < 3:
                logger.warning(
                    f"Only {n_files} flat frames found for setup: {key} (window: {value['window']}, x_bin: {value['bin_x']}, y_bin: {value['bin_y']}, filter: {value['filter']}), master flat will not be optimal..."
                )

            logger.info(
                f"Making master flat for setup: {key} (window: {value['window']}, x_bin: {value['bin_x']}, y_bin: {value['bin_y']}, filter: {value['filter']}) with {n_files} flat frames"
            )

            #
            skip_bias_correction = skip_bias_correction_input
            skip_dark_correction = skip_dark_correction_input

            dark_frame = None
            bias_frame = None

            if science_to_bias_map is None or science_to_bias_map[key] is None:
                logger.warning(
                    f"No bias key mapping found for flat setup key {key} in science_to_bias_map. Proceeding without dark or bias correction for flat frames in this setup."
                )
                skip_dark_correction = True
                skip_bias_correction = True

            # try loading dark frame if none are provided
            if dark_frames is None and not skip_dark_correction:
                dark_file_name = f"master_dark_{key}.fits"
                dark_frame = read_frame(output_dir, dark_file_name, self, logger)
                if dark_frame is None:
                    logger.warning(
                        f"No master dark found for key {key}. Proceeding without dark correction for flat frames in this setup."
                    )
                    skip_dark_correction = True
                else:
                    logger.info(
                        f"Successfully loaded master dark for key {key} from disk. Will apply dark correction to flat frames in this setup."
                    )

            # try loading bias frame if none are provided
            if bias_frames is None and not skip_bias_correction:
                bias_key = science_to_bias_map[key]
                bias_file_name = f"master_bias_{bias_key}.fits"
                bias_frame = read_frame(output_dir, bias_file_name, self, logger)
                if bias_frame is None:
                    logger.warning(
                        f"No master bias found for key {bias_key} mapped from flat setup key {key}. Proceeding without bias correction for flat frames in this setup."
                    )
                    skip_bias_correction = True
                else:
                    logger.info(
                        f"Successfully loaded master bias for key {bias_key} mapped from flat setup key {key} from disk. Will apply bias correction to flat frames in this setup."
                    )

            if (not skip_dark_correction) or (not skip_bias_correction):
                key_to_bias = science_to_bias_map[key]

            try:
                if dark_frame is None and not skip_dark_correction:
                    dark_frame = dark_frames[key_to_bias]
            except KeyError:
                logger.warning(
                    f"No master dark found for key {key_to_bias} mapped from flat setup key {key} in provided dark_frames. Proceeding without dark correction for flat frames in this setup."
                )
                skip_dark_correction = True

            if bias_frame is None and not skip_bias_correction:
                try:
                    bias_frame = bias_frames[key_to_bias]
                except KeyError:
                    logger.warning(
                        f"No master bias found for key {key_to_bias} mapped from flat setup key {key} in provided bias_frames. Proceeding without bias correction for flat frames in this setup."
                    )
                    skip_bias_correction = True

            flat_stack = []

            for file in value["files"]:

                filepath = os.path.join(input_dir, file)

                hdul = open_fits_file(filepath, logger)

                # create CCDData directly from the numpy array and keep header/meta
                try:
                    data = hdul[self.data_hdu_extension].data
                    hdr = (
                        hdul[self.data_hdu_extension].header
                        if len(hdul) > self.data_hdu_extension
                        else None
                    )
                    ccd_data = CCDData(
                        data,
                        unit=u.adu,
                        meta={"header": hdr} if hdr is not None else None,
                    )

                except Exception:
                    # fallback: create CCDData without header
                    ccd_data = CCDData(hdul[self.data_hdu_extension].data, unit=u.adu)

                if not skip_dark_correction:

                    logger.info(
                        f"Applying dark correction to flat frame {file} using master dark for key {key_to_bias} mapped from flat setup key {key}."
                    )

                    ccd_data = subtract_dark(ccd_data, dark_frame)

                if not skip_bias_correction:

                    logger.info(
                        f"Applying bias correction to flat frame {file} using master bias for key {key_to_bias} mapped from flat setup key {key}."
                    )

                    ccd_data = subtract_bias(ccd_data, bias_frame)

                flat_stack.append(ccd_data)

            # 2 sigma clipped median master flat
            master_flat = combine(
                flat_stack,
                method="median",
                sigma_clip=True,
                sigma_clip_low_thresh=2,
                sigma_clip_high_thresh=2,
            )

            bpm_copy = (
                bad_pixel_masks[key_to_bias].copy().astype(bool)
                if bad_pixel_masks is not None and key_to_bias in bad_pixel_masks
                else None
            )

            # make a seperate bpm mask for every science configuration so
            # we don't end up overriding the same one
            bad_pixel_mask = self.update_bad_pixel_map(
                master_flat.data,
                logger,
                output_dir,
                key,
                bpm_copy,
                show_plots=show_plots,
            )

            bad_pixel_masks_science[key] = bad_pixel_mask

            # for displaying and normalazation
            masked_flat = np.ma.masked_array(master_flat.data, mask=bad_pixel_mask)

            masked_median = np.ma.median(masked_flat)

            masked_normalized = masked_flat / masked_median
            master_normalized = master_flat / masked_median

            plt.close()

            plt.imshow(masked_normalized, cmap="gray", origin="lower")

            plt.colorbar()
            plt.title(
                f"Master Flat for window: {value['window']}, x_bin: {value['bin_x']}, y_bin: {value['bin_y']}, filter: {value['filter']}"
            )

            # save plot to the output directory
            plot_path = os.path.join(output_dir, f"master_flat_{key}.png")
            plt.savefig(plot_path)

            if show_plots:
                plt.show()

            # open the middle frame, copy its HDUList, replace the data with master_bias and write to disk
            mid_idx = len(value["files"]) // 2
            mid_file = value["files"][mid_idx]
            mid_path = os.path.join(input_dir, mid_file)

            hdul_mid = open_fits_file(mid_path, logger)

            # deep-copy all HDUs so we don't mutate the original HDUList in memory
            hdul_copy = fits.HDUList([hdu.copy() for hdu in hdul_mid])

            header_updates = {
                "MASTERFLAT": (True, "Indicates this frame is a master flat"),
                "FLATCNT": (len(value["files"]), "Number of flat frames combined"),
                "FLATWIN": (value["window"], "Window setting for this master flat"),
                "FLATBINX": (value["bin_x"], "Binning in X for this master flat"),
                "FLATBINY": (value["bin_y"], "Binning in Y for this master flat"),
            }

            # also write to header updates whether dark or and bias corrected
            if not skip_dark_correction:
                header_updates["DARKCORR"] = (
                    True,
                    "Indicates dark correction applied to flat frames",
                )
                header_updates["DARKKEY"] = (
                    key_to_bias,
                    "Key of the master dark used for correction",
                )
            else:
                header_updates["DARKCORR"] = (
                    False,
                    "Indicates dark correction was not applied to flat frames",
                )

            if not skip_bias_correction:
                header_updates["BIASCORR"] = (
                    True,
                    "Indicates bias correction applied to flat frames",
                )
                header_updates["BIASKEY"] = (
                    key_to_bias,
                    "Key of the master bias used for correction",
                )
            else:
                header_updates["BIASCORR"] = (
                    False,
                    "Indicates bias correction was not applied to flat frames",
                )

            write_frame(
                self,
                hdul_copy,
                master_normalized.data,
                f"master_flat_{key}.fits",
                output_dir,
                logger,
                bad_pixel_mask=bad_pixel_mask,
                comment=f"Master flat created on {datetime.now().isoformat()} using {len(value['files'])} flat frames with 2-sigma clipping. Bad pixel mask updated based on deviation from median and zero-value pixels.",
                header_updates=header_updates,
            )

            master_flats[key] = master_flat

            logger.info("Master flat creation complete.")

        return master_flats, bad_pixel_masks

    def reduce_science_frames(
        self,
        raw_data_path,
        output_dir,
        science_configurations,
        logger,
        show_plots=False,
        bad_pixel_masks=None,
        dark_frames=None,
        bias_frames=None,
        flat_frames=None,
        science_to_bias_map=None,
        skip_dark=False,
        skip_bias=False,
        skip_flats=False,
    ):

        # These booleans control whether to skip certain steps
        # The copy is taken to reset at every configuration, since
        # the logic flow might differ
        skip_dark_input = skip_dark
        skip_bias_input = skip_bias
        skip_flats_input = skip_flats

        for key, value in science_configurations.items():
            logger.info(
                f"Reducing science frames for setup: {key} (window: {value['window']}, x_bin: {value['bin_x']}, y_bin: {value['bin_y']}, filter: {value['filter']}) with {len(value['files'])} science frames"
            )

            # reset the bools for this configuration
            skip_dark = skip_dark_input
            skip_bias = skip_bias_input
            skip_flats = skip_flats_input

            # TODO - generalize loading masters into one method
            # load dark frames from disc if not provided and not skipped
            if not skip_dark and dark_frames is None:
                # same as bias key
                dark_key = science_to_bias_map[key]
                dark_file_name = f"master_dark_{dark_key}.fits"
                dark_frame = read_frame(output_dir, dark_file_name, self, logger)
                if dark_frame is not None:
                    logger.info(
                        f"Successfully loaded master dark for key {key} from disk for science reduction."
                    )
                else:
                    logger.warning(
                        f"No master dark found for key {key} in science_to_bias_map. Science reduction for setups with this bias key mapping will proceed without dark correction."
                    )

                    skip_dark = True

            elif science_to_bias_map is None:
                logger.warning(
                    f"No bias key mapping found for science setup key {key} in science_to_bias_map. Proceeding without dark correction for science frames in this setup."
                )
                skip_dark = True

            else:
                dark_frame = dark_frames[dark_key]

            # load bias frames from disc if not provided and not skipped
            if not skip_bias and bias_frames is None:
                bias_key = science_to_bias_map[key]
                bias_file_name = f"master_bias_{bias_key}.fits"
                bias_frame = read_frame(output_dir, bias_file_name, self, logger)
                if bias_frame is not None:
                    logger.info(
                        f"Successfully loaded master bias for key {bias_key} mapped from science setup key {key} from disk for science reduction."
                    )
                else:
                    logger.warning(
                        f"No master bias found for key {bias_key} mapped from science setup key {key} in science_to_bias_map. Science reduction for setups with this bias key mapping will proceed without bias correction."
                    )

                    skip_bias = True

            elif science_to_bias_map is None:
                logger.warning(
                    f"No bias key mapping found for science setup key {key} in science_to_bias_map. Proceeding without bias correction for science frames in this setup."
                )
                skip_bias = True

            else:
                bias_frame = bias_frames[bias_key]

            # load flat frames from disc if not provided and not skipped
            if not skip_flats and flat_frames is None:
                flat_key = key
                flat_file_name = f"master_flat_{flat_key}.fits"
                flat_frame = read_frame(output_dir, flat_file_name, self, logger)
                if flat_frame is not None:
                    logger.info(
                        f"Successfully loaded master flat for key {flat_key} from disk for science reduction."
                    )
                else:
                    logger.warning(
                        f"No master flat found for key {flat_key} in science_to_bias_map. Science reduction for setups with this bias key mapping will proceed without flat correction."
                    )

                    skip_flats = True

            elif science_to_bias_map is None:
                logger.warning(
                    f"No bias key mapping found for science setup key {key} in science_to_bias_map. Proceeding without flat correction for science frames in this setup."
                )
                skip_flats = True

            else:
                flat_frame = flat_frames[flat_key]

            for file in value["files"]:

                filepath = os.path.join(raw_data_path, file)

                hdul = open_fits_file(filepath, logger)

                # create CCDData directly from the numpy array and keep header/meta
                try:
                    data = hdul[self.data_hdu_extension].data
                    hdr = (
                        hdul[self.data_hdu_extension].header
                        if len(hdul) > self.data_hdu_extension
                        else None
                    )
                    ccd_data = CCDData(
                        data,
                        unit=u.adu,
                        meta={"header": hdr} if hdr is not None else None,
                    )

                except Exception:
                    # fallback: create CCDData without header
                    ccd_data = CCDData(hdul[self.data_hdu_extension].data, unit=u.adu)

                if not skip_dark:

                    logger.info(
                        f"Applying dark correction to science frame {file} using master dark for key {dark_key} mapped from science setup key {key}."
                    )

                    ccd_data = subtract_dark(ccd_data, dark_frame.data)

                if not skip_bias:

                    logger.info(
                        f"Applying bias correction to science frame {file} using master bias for key {bias_key} mapped from science setup key {key}."
                    )

                    ccd_data = subtract_bias(ccd_data, bias_frame.data)

                if not skip_flats:

                    logger.info(
                        f"Applying flat correction to science frame {file} using master flat for key {flat_key} mapped from science setup key {key}."
                    )

                    ccd_data = flat_correct(ccd_data, flat_frame.data)

                if not skip_flats:
                    bad_pixel_mask = flat_frame.bpm

                elif not skip_bias:
                    bad_pixel_mask = bias_frame.bpm

                else:
                    bad_pixel_mask = None

                detector_array = np.array(ccd_data.data.copy())
                if bad_pixel_mask is not None:
                    detector_array = np.ma.masked_array(
                        detector_array, mask=bad_pixel_mask
                    )

                min_percentile = np.nanpercentile(ccd_data.data, 30)
                max_percentile = np.nanpercentile(ccd_data.data, 95)

                plt.close()
                plt.imshow(
                    detector_array,
                    cmap="gray",
                    origin="lower",
                    vmin=min_percentile,
                    vmax=max_percentile,
                )
                plt.colorbar()
                plt.title(
                    f"Reduced Science Frame for {file}, dark corrected: {not skip_dark}, bias corrected: {not skip_bias}, flat corrected: {not skip_flats}"
                )
                plt.tight_layout()

                save_path = os.path.join(
                    output_dir, f"reduced_science_{file.split('.')[0]}.png"
                )
                plt.savefig(save_path)

                if show_plots:
                    plt.show()
                else:
                    plt.close()

                # write to disc

                hdul_copy = fits.HDUList([hdu.copy() for hdu in hdul])

                write_frame(
                    self,
                    hdul_copy,
                    ccd_data.data,
                    f"reduced_science_{file}",
                    output_dir,
                    logger,
                    bad_pixel_mask=bad_pixel_mask,
                    comment=f"Science frame reduced on {datetime.now().isoformat()} with dark correction: {not skip_dark}, bias correction: {not skip_bias}, flat correction: {not skip_flats}.",
                    header_updates={
                        "REDUCTION": (True, "Indicates this frame has been reduced"),
                        "DARKCORR": (
                            not skip_dark,
                            "Indicates whether dark correction was applied",
                        ),
                        "BIASCORR": (
                            not skip_bias,
                            "Indicates whether bias correction was applied",
                        ),
                        "FLATCORR": (
                            not skip_flats,
                            "Indicates whether flat correction was applied",
                        ),
                        "DARKKEY": (
                            dark_key if not skip_dark else None,
                            "Key of the master dark used for correction, if applicable",
                        ),
                        "BIASKEY": (
                            bias_key if not skip_bias else None,
                            "Key of the master bias used for correction, if applicable",
                        ),
                        "FLATKEY": (
                            flat_key if not skip_flats else None,
                            "Key of the master flat used for correction, if applicable",
                        ),
                    },
                )


class ALFOSC(Instrument):
    """ALFOSC instrument configuration for NOT telescope"""

    def __init__(self):
        # Define ALFOSC CCD detector parameters
        alfosc_ccd = Detector(
            window_keyword=("DETWIN1", 0),
            bin_x_keyword=("DETXBIN", 0),
            bin_y_keyword=("DETYBIN", 0),
            bpm_median_threshold=0.25,
        )

        # Initialize parent class with ALFOSC-specific parameters
        super().__init__(
            name="ALFOSC",
            detector=alfosc_ccd,
            data_hdu_extension=1,
            filter_keyword=(["ALFLTNM", "FAFLTNM", "FBFLTNM"], 0),
            obsmode_keyword=("OBS_MODE", 0),
            imaging_obsmode_keyword=("IMAGING", 0),
            imagetype_keyword=("IMAGETYP", 0),
            bias_keyword=["BIAS"],
            dark_keyword=["DARK"],
            flat_keyword=["FLAT,SKY"],
            science_keyword=["OBJECT"],
            object_keyword=("OBJECT", 0),
        )

    def match_image_type(self, hdul) -> Optional[ImageType]:

        if hdul[0].header["IMAGETYP"] in self.bias_keyword:
            return ImageType.BIAS

        if (
            hdul[0].header["IMAGETYP"] in self.flat_keyword
            and hdul[0].header["ALAPRTNM"] == "Open"
        ):
            return ImageType.FLAT

        if (
            hdul[0].header["IMAGETYP"] in self.science_keyword
            and hdul[0].header["ALAPRTNM"] == "Open"
            and hdul[0].header["OBS_MODE"] == "IMAGING"
        ):
            return ImageType.SCIENCE

    def make_master_flat(
        self,
        input_dir,
        output_dir,
        flat_setup,
        logger,
        bad_pixel_masks=None,
        dark_frames=None,
        bias_frames=None,
        science_to_bias_map=None,
        show_plots=False,
        skip_dark_correction=True,
        skip_bias_correction=False,
    ):
        return super().make_master_flat(
            input_dir,
            output_dir,
            flat_setup,
            logger,
            bad_pixel_masks,
            dark_frames,
            bias_frames,
            science_to_bias_map,
            show_plots,
            skip_dark_correction,
            skip_bias_correction,
        )

    def reduce_science_frames(
        self,
        raw_data_path,
        output_dir,
        science_configurations,
        logger,
        show_plots=False,
        bad_pixel_masks=None,
        dark_frames=None,
        bias_frames=None,
        flat_frames=None,
        science_to_bias_map=None,
        skip_dark=True,
        skip_bias=False,
        skip_flats=False,
    ):
        return super().reduce_science_frames(
            raw_data_path,
            output_dir,
            science_configurations,
            logger,
            show_plots=show_plots,
            bad_pixel_masks=bad_pixel_masks,
            dark_frames=dark_frames,
            bias_frames=bias_frames,
            flat_frames=flat_frames,
            science_to_bias_map=science_to_bias_map,
            skip_dark=skip_dark,
            skip_bias=skip_bias,
            skip_flats=skip_flats,
        )


class NOTCAM(Instrument):
    """NOTCAM instrument configuration for NOT telescope"""

    def __init__(self):
        # Define NOTCAM CCD detector parameters
        notcam_swir3 = Detector(
            window_keyword=None,
            bin_x_keyword=None,
            bin_y_keyword=None,
            bpm_median_threshold=0.4,
        )

        # Initialize parent class with ALFOSC-specific parameters
        super().__init__(
            name="NOTCAM",
            detector=notcam_swir3,
            data_hdu_extension=1,
            filter_keyword=(["NCFLTNM1", "NCFLTNM2"], 0),
            obsmode_keyword=("OBS_MODE", 0),
            imaging_obsmode_keyword=("IMAGING", 0),
            imagetype_keyword=("IMAGETYP", 0),
            bias_keyword=None,
            dark_keyword=["DARK"],
            flat_keyword=["FLAT,SKY"],
            science_keyword=["OBJECT"],
            exposure_time_keyword=("EXPTIME", 1),
            object_keyword=("OBJECT", 0),
        )

    def get_header_value(self, hdul, keyword_tuple) -> Optional[str]:
        """Helper method to extract header value based on provided keyword tuple"""

        # these are all constant for NOTCAM, so we use a dummy value
        if keyword_tuple in [
            self.detector.window_keyword,
            self.detector.bin_x_keyword,
            self.detector.bin_y_keyword,
        ]:
            return "0"

        else:
            return super().get_header_value(hdul, keyword_tuple)

    def match_image_type(self, hdul) -> Optional[ImageType]:

        # bias frames are not applicable for NOTCAM, so we skip that check

        if "skyflat" in hdul[0].header["OBJECT"]:
            return ImageType.FLAT

        if ("dark" in hdul[0].header["OBJECT"]) or ("dfra" in hdul[0].header["OBJECT"]):
            return ImageType.DARK

        if (
            ("OBJECT" in hdul[0].header["IMAGETYP"])
            and (hdul[0].header["IMAGECAT"] == "SCIENCE")
            and (hdul[0].header["NCGRNM"] == "Open")
        ):
            return ImageType.SCIENCE

    def make_master_bias(
        self,
        input_dir,
        output_dir,
        bias_setup,
        logger,
        bad_pixel_masks=None,
        show_plots=False,
    ):
        """
        Not needed for NOTCAM.
        """
        return None, None

    def make_master_flat(
        self,
        input_dir,
        output_dir,
        flat_setup,
        logger,
        bad_pixel_masks=None,
        dark_frames=None,
        bias_frames=None,
        science_to_bias_map=None,
        show_plots=False,
    ):
        """
        For NOTcam, differential flats are used, so we do some
        pre-processing before doing the "standard" master flat creation.
        """

        for key, value in flat_setup.items():

            logger.info(
                f"Pre-processing NOTCAM flats for setup: {key} (window: {value['window']}, x_bin: {value['bin_x']}, y_bin: {value['bin_y']}, filter: {value['filter']}) with {len(value['files'])} flat frames"
            )

            raw_data_list = np.array([])
            intensity_medians = []

            # use Python lists to collect arrays and medians (avoids flattening with np.append)
            raw_data_list = []
            intensity_medians = []
            exptimes = []
            object_names = []

            for file in value["files"]:

                # first load all frames and sort them by median intensity

                filepath = os.path.join(input_dir, file)

                hdul = open_fits_file(filepath, logger)

                try:
                    data = hdul[self.data_hdu_extension].data
                    exptime = get_header_value(hdul, self.exposure_time_keyword, logger)
                    object_name = get_header_value(hdul, self.object_keyword, logger)

                except Exception:
                    logger.error(
                        f"Error reading data from HDU {self.data_hdu_extension} in file {filepath}. Skipping this file for master flat creation for NOTcam."
                    )
                    continue

                raw_data_list.append(np.array(data))

                # compute the median intensity of the current frame
                median_intensity = np.median(data)
                intensity_medians.append(median_intensity)
                exptimes.append(exptime)
                object_names.append(object_name)

            # sort by median intensity using argsort (descending) and reorder lists
            meds = np.asarray(intensity_medians)
            if meds.size == 0:
                continue

            order = np.argsort(meds)[::-1]  # indices for descending medians

            # reorder filenames
            value["files"] = [value["files"][i] for i in order]
            # assume raw_data_list already aligns with value["files"]
            raw_data_list = [raw_data_list[i] for i in order]
            intensity_medians = np.array([float(meds[i]) for i in order])
            exptimes = np.array([float(exptimes[i]) for i in order])
            object_names = np.array([object_names[i] for i in order])

            # find the most used exptime and exclude all others
            used_exptime = np.bincount(exptimes.astype(int)).argmax()
            logger.info(f"Using exposure time {used_exptime} for master flat creation.")

            # exclude all frames with different exposure times
            mask = exptimes == used_exptime
            value["files"] = [value["files"][i] for i in range(len(mask)) if mask[i]]
            raw_data_list = [raw_data_list[i] for i in range(len(mask)) if mask[i]]
            intensity_medians = intensity_medians[mask]
            exptimes = exptimes[mask]
            object_names = [object_names[i] for i in range(len(mask)) if mask[i]]

            # NOTcam specific clean-up from manual flats:
            # For the brightests flats, see if several frames have the object
            # "skyflat 1" and keep only the last one:

            skyflat_mask = [name == "skyflat 1" for name in object_names]
            remove_mask = np.zeros(len(skyflat_mask), dtype=bool)

            if (
                sum(skyflat_mask) > 2
            ):  # booth first bright and dark skyflat have "skyflat 1" as object
                logger.info(
                    f"Found multiple 'skyflat 1' frames for config {key}. Keeping only the last one."
                )
                for i in range(len(skyflat_mask) - 1):
                    if skyflat_mask[i] and skyflat_mask[i + 1]:
                        remove_mask[i] = True

                # apply the remove mask to all lists
                value["files"] = [
                    value["files"][i]
                    for i in range(len(remove_mask))
                    if not remove_mask[i]
                ]
                raw_data_list = [
                    raw_data_list[i]
                    for i in range(len(remove_mask))
                    if not remove_mask[i]
                ]
                intensity_medians = intensity_medians[~remove_mask]
                exptimes = exptimes[~remove_mask]
                object_names = [
                    object_names[i]
                    for i in range(len(remove_mask))
                    if not remove_mask[i]
                ]

            plt.close()
            plt.plot(value["files"], intensity_medians, marker="o")
            plt.xlabel("File")
            plt.xticks(rotation=90, ha="right")
            plt.ylabel("Median Intensity")
            plt.title(
                f"NOTCAM-flats Median Intensities for conf. key: {key}.\n Should be in descending order, with bright flats first and dark flats last."
            )
            plt.tight_layout()
            plot_save_path = os.path.join(output_dir, f"notcam_flat_medians_{key}.png")
            plt.savefig(plot_save_path)
            if show_plots:
                plt.show()

            # check that there is an even equal number of frames to
            # make pairs

            bright_frames = raw_data_list[0 : len(raw_data_list) // 2]
            dark_frames = raw_data_list[len(raw_data_list) // 2 :]

            bright_medians = intensity_medians[0 : len(intensity_medians) // 2]
            dark_medians = intensity_medians[len(intensity_medians) // 2 :]

            diff_medians = bright_medians - dark_medians

            # make sure the median differneces are positive and descending,
            # otherwise throw a warning
            if not np.all(diff_medians > 0):
                logger.warning(
                    f"Found non-positive median differences between bright and dark NOTCAM flats for config {key}. This might indicate an issue with the flat frames or the pairing process."
                )

            # also check that the differences are in descending order, otherwise throw a warning
            if not np.all(diff_medians[:-1] >= diff_medians[1:]):
                logger.warning(
                    f"Found non-descending median differences between bright and dark NOTCAM flats for config {key}. This might indicate an issue with the flat frames or the pairing process."
                )

            diff_frames = [
                bright_frames[i] - dark_frames[i] for i in range(len(bright_frames))
            ]
            diff_frame_medians = [np.median(frame) for frame in diff_frames]

            plt.close()
            plt.plot(diff_frame_medians, marker="o")
            plt.title(
                f"Medians of separate differential NOTCAM flats for conf. key: {key}"
            )
            plt.xlabel("Frame Index")
            plt.ylabel("Median Intensity")
            plt.tight_layout()
            plot_save_path = os.path.join(
                output_dir, f"notcam_flat_diff_medians_{key}.png"
            )
            plt.savefig(plot_save_path)
            if show_plots:
                plt.show()

            logger.info(
                "Scaling differential NOTCAM flats by their median intensity to account for brightness differences between pairs of bright and dark flats."
            )

            diff_frames_scaled = [
                diff_frames[i] / diff_frame_medians[i] for i in range(len(diff_frames))
            ]

            # write the frames to output directory, name them diff_filename, and update
            # the flat setup so it can be passed to the standard master flat creation method
            # TODO this is a bit hacky, refine later
            diff_file_names = []
            for i, diff in enumerate(diff_frames_scaled):
                diff_file_name = f"diff_{key}_{i}.fits"
                diff_file_path = os.path.join(input_dir, diff_file_name)

                # place the diff data in the configured data HDU extension
                if self.data_hdu_extension == 0:
                    hdu = fits.PrimaryHDU(diff)
                    hdul = fits.HDUList([hdu])
                else:
                    # create a primary HDU and pad with empty image HDUs until the desired extension
                    primary = fits.PrimaryHDU()
                    hdus = [primary]
                    # create placeholder image HDUs to fill gaps (use zeros with same shape/dtype)
                    for _ in range(self.data_hdu_extension - 1):
                        hdus.append(fits.ImageHDU(np.zeros_like(diff)))
                    # append the real data at the configured extension
                    hdus.append(fits.ImageHDU(diff))
                    hdul = fits.HDUList(hdus)

                hdul.writeto(diff_file_path, overwrite=True)

                diff_file_names.append(diff_file_name)
                logger.info(
                    f"Wrote differential flat {diff_file_name} to {diff_file_path}"
                )

            # update the flat setup to point to the newly written differential files
            value["files"] = diff_file_names

        return super().make_master_flat(
            input_dir,
            output_dir,
            flat_setup,
            logger,
            show_plots=show_plots,
            skip_bias_correction=True,
            skip_dark_correction=True,
        )
