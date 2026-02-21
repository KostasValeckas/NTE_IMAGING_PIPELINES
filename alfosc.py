from instruments import Detector, Telescope, Instrument
from reduction import ReductionPipeline
import sys
from pathlib import Path

alfosc_ccd = Detector(
    window_keyword = ("DETWIN1", 0),
    bin_x_keyword = ("DETXBIN", 0),
    bin_y_keyword = ("DETYBIN", 0),
)


alfosc = Instrument(

    detector = alfosc_ccd,

    data_hdu_extension= 1,

    filter_keyword = (["ALFLTNM", "FAFLTNM", "FBFLTNM"], 0),

    obsmode_keyword = (("OBS_MODE", 0)),
    imaging_obsmode_keyword = (("IMAGING", 0)),
    imagetype_keyword = (("IMAGETYP", 0)),
    bias_keyword = (["BIAS"]),
    dark_keyword = (["DARK"]),
    flat_keyword = (["FLAT,SKY"]),
    science_keyword = (["OBJECT"]),
)


if __name__ == "__main__":

    # TODO: should logger be initialized already here?

    if len(sys.argv) != 2:
        print(f"Usage: {Path(sys.argv[0]).name} RAW_DATA_PATH", file=sys.stderr)
        sys.exit(1)

    raw_data_path = Path(sys.argv[1])

    if not raw_data_path.exists():
        print(f"Error: raw data path does not exist: {raw_data_path}", file=sys.stderr)
        sys.exit(1)

    pipeline = ReductionPipeline(alfosc, raw_data_path)

    pipeline.run_pipeline()
