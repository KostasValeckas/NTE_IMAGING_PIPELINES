from dataclasses import dataclass
from typing import Optional
from enum import Enum


class ImageType(Enum):
    BIAS = "BIAS"
    DARK = "DARK"
    FLAT = "FLAT"
    SCIENCE = "SCIENCE"


@dataclass
class FitsHeaderEntry:
    key: str
    value: Optional[str] = None
    comment: Optional[str] = None


@dataclass
class Detector:
    gain: Optional[tuple[str, int]] = None
    read_noise: Optional[tuple[str, int]] = None
    saturation_level: Optional[tuple[str, int]] = None
    dark_current: Optional[tuple[str, int]] = None
    pixel_scale: Optional[tuple[str, int]] = None
    field_of_view: Optional[tuple[str, int]] = None
    window_keyword: Optional[tuple[str, int]] = (None,)
    bin_x_keyword: Optional[tuple[str, int]] = (None,)
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
        data_hdu_extension: Optional[int] = None
    ):
        self.detector = detector if detector is not None else Detector()
        self.telescope = telescope if telescope is not None else Telescope()

        self.filter_keyword = filter_keyword if filter_keyword is not None else ([], 0)


        self.obsmode_keyword = obsmode_keyword if obsmode_keyword is not None else (None, 0)
        self.imaging_obsmode_keyword = (
            imaging_obsmode_keyword if imaging_obsmode_keyword is not None else (None, 0)
        )
        self.imagetype_keyword = (
            imagetype_keyword if imagetype_keyword is not None else (None, 0)
        )
        self.bias_keyword = bias_keyword if bias_keyword is not None else []
        self.dark_keyword = dark_keyword if dark_keyword is not None else []
        self.flat_keyword = flat_keyword if flat_keyword is not None else []
        self.science_keyword = science_keyword if science_keyword is not None else []


        self.data_hdu_extension = data_hdu_extension

    def match_image_type(self, hdul) -> Optional[ImageType]:
        
        # Base implementation: intended to be overridden by subclasses.
        # Return None to indicate "no determination" at this level.
        return None


    def get_header_value(self, hdul, keyword_tuple) -> Optional[str]:
        """Helper method to extract header value based on provided keyword tuple"""
        if keyword_tuple is None or len(keyword_tuple) != 2:
            return None
        
        keys, hdu_index = keyword_tuple
        if not isinstance(keys, list):
            keys = [keys]
        
        for key in keys:
            try:
                value = hdul[hdu_index].header[key]
                if value is not None:
                    return value
            except KeyError:
                continue
        
        return None
    

    def get_header_values(self, hdul, keyword_tuple) -> Optional[dict[str, Optional[str]]]:
        """Helper method to extract multiple header values based on provided keyword tuple"""
        if keyword_tuple is None or len(keyword_tuple) != 2:
            return None
        
        keys, hdu_index = keyword_tuple
        if not isinstance(keys, list):
            keys = [keys]
        
        values = []
        for key in keys:
            try:
                value = hdul[hdu_index].header[key]
                values.append(value)
            except KeyError:
                values.append(None)

        return values