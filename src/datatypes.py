from enum import Enum
from dataclasses import dataclass
from typing import Optional
from ccdproc import CCDData
import astropy.units as u

"""
A module started for creating custom datatypes, but not fully utilized yet.
The methods here are used in the code however, so do not remove them.
"""

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
    """
    Class created for easier handling of CCDDData objects from `ccdproc`.
    
    Parameters
    ----------
    hdul : astropy.io.fits.HDUList
        The HDUList object containing the FITS header and data.
    data : numpy.ndarray
        The image data as a 2D numpy array.
    bpm : numpy.ndarray
        The bad pixel mask as a 2D numpy array, where bad pixels are marked 
        with 1 and good pixels with 0.
    """
    def __init__(self, hdul, data, bpm):

        self.hdul = hdul
        self.data = CCDData(data, unit=u.adu)
        self.bpm = bpm
