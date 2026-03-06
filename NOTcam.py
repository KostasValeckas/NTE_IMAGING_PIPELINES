from instruments import Detector, Telescope, Instrument
from reduction import ReductionPipeline
import sys
from pathlib import Path
from typing import Optional
from instruments import ImageType


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
            name = "NOTCAM",
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
    

        #these are all constant for NOTCAM, so we use a dummy value
        if keyword_tuple in [self.detector.window_keyword, self.detector.bin_x_keyword, self.detector.bin_y_keyword]:
            return "0"

        else:
            return super().get_header_value(hdul, keyword_tuple)


    def match_image_type(self, hdul) -> Optional[ImageType]:

        # bias frames are not applicable for NOTCAM, so we skip that check
        
        if "skyflat" in hdul[0].header["OBJECT"]:
            return ImageType.FLAT

        if ("dark" in hdul[0].header["OBJECT"]) or ("dfra" in hdul[0].header["OBJECT"]):
            return ImageType.DARK

        if ("OBJECT" in hdul[0].header["IMAGETYP"]) and (hdul[0].header["IMAGECAT"] == "SCIENCE") and (hdul[0].header["NCGRNM"] == "Open"):
            return ImageType.SCIENCE
        

if __name__ == "__main__":

    # TODO: should logger be initialized already here?

    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} RAW_DATA_PATH", file=sys.stderr)
        sys.exit(1)

    raw_data_path = Path(sys.argv[1])

    if not raw_data_path.exists():
        print(f"Error: raw data path does not exist: {raw_data_path}", file=sys.stderr)
        sys.exit(1)

    # Create NOTCAM instrument instance
    notcam = NOTCAM()

    pipeline = ReductionPipeline(notcam, raw_data_path)

    pipeline.run_pipeline()
