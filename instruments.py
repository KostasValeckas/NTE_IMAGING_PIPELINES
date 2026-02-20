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
    gain: Optional[float] = None
    read_noise: Optional[float] = None
    saturation_level: Optional[float] = None
    dark_current: Optional[float] = None
    pixel_scale: Optional[float] = None
    field_of_view: Optional[float] = None


@dataclass
class Telescope:
    name: Optional[str] = None
    aperture: Optional[float] = None
    focal_length: Optional[float] = None
    location: Optional[str] = None


class Instrument:
    def __init__(
        self,
        detector: Optional[Detector] = None,
        telescope: Optional[Telescope] = None,
        imagetype_keyword: Optional[str] = None,
        bias_keyword: Optional[str] = None,
        dark_keyword: Optional[str] = None,
        flat_keyword: Optional[str] = None,
        science_keyword: Optional[str] = None,
    ):
        self.detector = detector if detector is not None else Detector()
        self.telescope = telescope if telescope is not None else Telescope()

        self.imagetype_keyword = imagetype_keyword
        self.bias_keyword = bias_keyword
        self.dark_keyword = dark_keyword
        self.flat_keyword = flat_keyword
        self.science_keyword = science_keyword

    def match_image_type(self):

        match self.imagetype_keyword:
            case self.bias_keyword:
                return ImageType.BIAS
            case self.dark_keyword:
                return ImageType.DARK
            case self.flat_keyword:
                return ImageType.FLAT
            case self.science_keyword:
                return ImageType.SCIENCE
            case _:
                return None
