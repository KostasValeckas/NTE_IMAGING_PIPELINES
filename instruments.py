from dataclasses import dataclass
from typing import Optional
import os
from enum import Enum
from datatypes import ImageType
from ccdproc import combine, CCDData, subtract_dark, subtract_bias, flat_correct
from IO import open_fits_file, write_frame, read_frame
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
from astropy.io import fits
from datetime import datetime


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

        self.data_hdu_extension = data_hdu_extension

    def match_image_type(self, hdul) -> Optional[ImageType]:

        # Base implementation: intended to be overridden by subclasses.
        # Return None to indicate "no determination" at this level.
        return None

    def update_bad_pixel_map(
        self,
        master_frame,
        logger,
        bad_pixel_mask=None,
        n_median_deviation=0.2,
        show_plots=False,
    ):

        
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
        new_bad_pixels = np.abs(master_frame_masked - median) > (n_median_deviation * median)

        all_bad_pixels = zero_pixels | new_bad_pixels

        # combine with existing bad pixel mask

        combined_bad_pixel_mask = bad_pixel_mask | all_bad_pixels

        logger.info(f"Identified {np.sum(all_bad_pixels)} new bad pixels.")

        masked_array = np.ma.masked_array(master_frame_masked, mask=combined_bad_pixel_mask)

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

        if show_plots:
            plt.show()
        else:
            plt.close()

        return np.asarray(combined_bad_pixel_mask, dtype =bool)

    def make_master_bias(
        self,
        input_dir,
        output_dir,
        bias_setup,
        logger,
        bad_pixel_masks=None,
        show_plots=False
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
    ):
        """
        The bad pixel masks are assumed to be the one used from the dark and
        bias configurations only.
        """

        bad_pixel_masks_science = {}

        master_flats = {}

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

            # load the dark and bias masters

            skip_bias_correction = False
            skip_dark_correction = False

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

            key_to_bias = science_to_bias_map[key]

            # Mainly for debugging - if this is still None abort
            if key_to_bias == None:
                logger.error(
                    f"Science_to_bias_map entry for flat setup key {key} is None."
                    "Contact developers"
                )
                exit(-1)

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

            bpm_copy = bad_pixel_masks[key_to_bias].copy().astype(bool)


            # make a seperate bpm mask for every science configuration so
            # we don't end up overriding the same one
            bad_pixel_mask = self.update_bad_pixel_map(
                master_flat.data,
                logger,
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
        bad_pixel_masks = None,
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

            #TODO - generalize loading masters into one method
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
                    detector_array = np.ma.masked_array(detector_array, mask=bad_pixel_mask)

                min_percentile = np.nanpercentile(ccd_data.data, 30)
                max_percentile = np.nanpercentile(ccd_data.data, 95)

                plt.close()
                plt.imshow(detector_array, cmap="gray", origin="lower", vmin=min_percentile, vmax=max_percentile)
                plt.colorbar()
                plt.title(
                    f"Reduced Science Frame for {file}, dark corrected: {not skip_dark}, bias corrected: {not skip_bias}, flat corrected: {not skip_flats}"
                )
                plt.tight_layout()

                save_path = os.path.join(output_dir, f"reduced_science_{file.split('.')[0]}.png")
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
                        "DARKCORR": (not skip_dark, "Indicates whether dark correction was applied"),
                        "BIASCORR": (not skip_bias, "Indicates whether bias correction was applied"),
                        "FLATCORR": (not skip_flats, "Indicates whether flat correction was applied"),
                        "DARKKEY": (dark_key if not skip_dark else None, "Key of the master dark used for correction, if applicable"),
                        "BIASKEY": (bias_key if not skip_bias else None, "Key of the master bias used for correction, if applicable"),
                        "FLATKEY": (flat_key if not skip_flats else None, "Key of the master flat used for correction, if applicable"),
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
        
    def make_master_bias(self, **dummy_args):
        """
        Not needed for NOTCAM.
        """
        return None
        