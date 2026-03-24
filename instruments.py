from dataclasses import dataclass
from typing import Optional
import os
from enum import Enum
from datatypes import ImageType
from ccdproc import Combiner, CCDData 
from IO import open_fits_file
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u


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

    def make_master_bias(self, input_dir, bias_files, logger, bad_pixel_mask = None):

        bias_stack = []

        for file in bias_files:

            filepath = os.path.join(input_dir, file)

            hdul = open_fits_file(filepath, logger)

            # create CCDData directly from the numpy array and keep header/meta
            try:
                data = hdul[self.data_hdu_extension].data
                hdr = hdul[self.data_hdu_extension].header if len(hdul) > self.data_hdu_extension else None
                ccd_data = CCDData(data, unit=u.adu, meta={"header": hdr} if hdr is not None else None)
            except Exception:
                # fallback: create CCDData without header
                ccd_data = CCDData(hdul[self.data_hdu_extension].data, unit=u.adu)
            bias_stack.append(ccd_data)

        # 2 sigma clipped median master bias
        combiner = Combiner(bias_stack)
        combiner.sigma_clipping(low_thresh=2, high_thresh=2, func=np.ma.median)

        master_bias = combiner.median_combine()

        plt.imshow(master_bias.data, cmap="gray", vmin=np.percentile(master_bias.data, 5), vmax=np.percentile(master_bias.data, 95))
        plt.colorbar()
        plt.title("Master Bias")
        plt.show()
        


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




