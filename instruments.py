from dataclasses import dataclass
from typing import Optional
import os
from enum import Enum
from datatypes import ImageType
from ccdproc import combine, CCDData
from IO import open_fits_file
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
        self, master_frame, logger, bad_pixel_mask=None, n_median_deviation=0.2, show_plots=False
    ):

        # create a new bad pixel mask based on sigma clipping the master frame
        if bad_pixel_mask is None:
            bad_pixel_mask = np.zeros(master_frame.shape, dtype=bool)

        median = np.nanmedian(master_frame)


        zero_pixels = master_frame == 0
        logger.info(f"Identified {np.sum(zero_pixels)} zero-value pixels in master frame")


        logger.info(f"Masking pixels that deviate from median by more than {n_median_deviation} medians (median={median:.2f})")
        new_bad_pixels = np.abs(master_frame - median) > (n_median_deviation * median)

        all_bad_pixels = zero_pixels | new_bad_pixels

        # combine with existing bad pixel mask

        combined_bad_pixel_mask = bad_pixel_mask | all_bad_pixels

        logger.info(
            f"Identified {np.sum(all_bad_pixels)} new bad pixels."
        )

        masked_array = np.ma.masked_array(master_frame, mask=combined_bad_pixel_mask)

        # plot histogram of pixel values, excluding NaNs
        data_for_hist = masked_array.flatten()
        data_for_hist = data_for_hist[~np.isnan(data_for_hist)]
        n_bins = int(np.sqrt(data_for_hist.size))
        plt.figure()
        plt.hist(data_for_hist, bins=n_bins, color="gray", edgecolor="black")
        plt.xlabel("Pixel Value")
        plt.ylabel("Frequency")
        plt.title("Pixel Value Distribution - Everything outside the rejection thresholds is masked as bad pixel")

        # vertical lines for median and rejection thresholds
        plt.axvline(median, color="red", linestyle="-", linewidth=1.5, label=f"Median = {median:.2f}")
        lower = median - n_median_deviation * median
        upper = median + n_median_deviation * median
        plt.axvline(lower, color="orange", linestyle="--", linewidth=1.2, label=f"Reject < {lower:.2f}")
        plt.axvline(upper, color="orange", linestyle="--", linewidth=1.2, label=f"Reject > {upper:.2f}")
        plt.legend(loc="upper right")

        info_text = f"median={median:.2f}, rejection: |value - median| > {n_median_deviation}×median (±{n_median_deviation*median:.2f})"
        logger.info(info_text)

        # annotate plot with the same info
        plt.text(0.01, 0.95, info_text, transform=plt.gca().transAxes, fontsize=9, va="top")
        plt.grid()

        plt.xlim(median - n_median_deviation * median * 1.1, median + n_median_deviation * median * 1.1)

        if show_plots:
            plt.show()
        else:
            plt.close()

        return combined_bad_pixel_mask

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
                bad_pixel_masks[key] if key in bad_pixel_masks else None,
                show_plots=show_plots,
            )

            if key not in bad_pixel_masks:
                bad_pixel_masks[key] = bad_pixel_mask
            else:
                bad_pixel_masks[key] = bad_pixel_masks[key] | bad_pixel_mask

            plt.close()

            plt.imshow(
                master_bias.data,
                cmap="gray",
                vmin=np.percentile(master_bias.data, 5),
                vmax=np.percentile(master_bias.data, 95),
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

            # target HDU containing image data
            data_hdu = hdul_copy[self.data_hdu_extension]

            # directly assign master bias data (do not check or cast data types)
            data_hdu.data = master_bias.data


            #append the bad pixel mask as a new HDU to the HDUList
            bad_pixel_hdu = fits.ImageHDU(data=bad_pixel_mask.astype(np.uint8), name="BAD_PIXEL_MASK")
            bad_pixel_hdu.header["BPM_KEY"] = key
            hdul_copy.append(bad_pixel_hdu)

            # update header to record creation
            try:
                data_hdu.header["IMAGETYP"] = "MASTER_BIAS"
                data_hdu.header.add_history(f"Master bias created from stack {key}")
                data_hdu.header.add_history(
                    f"Created: {datetime.utcnow().isoformat()} UTC"
                )
            except Exception:
                # ignore header update errors
                pass

            # write master bias FITS to output directory
            out_path = os.path.join(output_dir, f"master_bias_{key}.fits")
            hdul_copy.writeto(out_path, overwrite=True)
            logger.info(f"Wrote master bias to {out_path}")

            # close HDULists
            try:
                hdul_mid.close()
            except Exception:
                pass
            try:
                hdul_copy.close()
            except Exception:
                pass

            master_biases[key] = master_bias

        logger.info("Master bias creation complete.")

        return master_biases, bad_pixel_masks



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
