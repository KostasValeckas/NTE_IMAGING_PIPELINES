from instruments import Detector, Instrument


from typing import Optional
from instruments import ImageType



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
            name = "ALFOSC",
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
            object_keyword=("OBJECT", 0)
        )

    def match_image_type(self, hdul) -> Optional[ImageType]:

        if hdul[0].header["IMAGETYP"] in self.bias_keyword:
            return ImageType.BIAS
        
        if hdul[0].header["IMAGETYP"] in self.flat_keyword and hdul[0].header["ALAPRTNM"] == "Open":
            return ImageType.FLAT
        
        if hdul[0].header["IMAGETYP"] in self.science_keyword and hdul[0].header["ALAPRTNM"] == "Open" and hdul[0].header["OBS_MODE"] == "IMAGING":
            return ImageType.SCIENCE
        


