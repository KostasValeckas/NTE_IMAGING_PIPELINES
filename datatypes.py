from enum import Enum
from dataclasses import dataclass
from typing import Optional
from ccdproc import CCDData
import astropy.units as u


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


class Processed_frame:

    def __init__(self, hdul, data, bpm):

        self.hdul = hdul
        self.data = CCDData(data, unit=u.adu)
        self.bpm = bpm
