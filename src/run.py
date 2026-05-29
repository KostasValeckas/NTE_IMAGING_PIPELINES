import argparse
import sys
from pathlib import Path
from instruments import *


from reduction import ReductionPipeline


def main():
    # TODO: should logger be initialized already here?

    parser = argparse.ArgumentParser(description="Run imaging pipeline")
    parser.add_argument("raw_data_path", type=Path, help="Path to raw data")
    parser.add_argument(
        "-out",
        "--output-dir",
        type=Path,
        help="Output directory for reduced files (default: RAW_DATA_PATH/reduced)",
        default=None,
    )
    parser.add_argument(
        "-o",
        "--override",
        action="store_true",
        help="Override existing output directory if it exists",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show debugging plots during reduction",
    )
    parser.add_argument(
        "instrument", type=str, help="Instrument to use. Options: ALFOSC, NOTcam"
    )

    parser.add_argument(
        "--skip-reduction",
        action="store_true",
        help="Skip reduction steps and only run photometric calibrations on existing reduced data",
    )

    parser.add_argument(
        "--skip-phot-calib",
        action="store_true",
        help="Skip photometric calibrations after reduction",
    )

    parser.add_argument(
        "--skip-WCS-refinement",
        action="store_true",
        help="Skip WCS refinement step during photometric calibrations",
    )

    args = parser.parse_args()

    raw_data_path = args.raw_data_path

    if not raw_data_path.exists():
        print(f"Error: raw data path does not exist: {raw_data_path}", file=sys.stderr)
        sys.exit(1)

    # set default output dir if not provided
    output_dir = args.output_dir

    if output_dir is not None:

        if not output_dir.exists():
            print(f"Output directory does not exist, creating: {output_dir}")
            output_dir.mkdir(parents=True, exist_ok=True)

    # Create instrument instance based on the provided name
    instrument_name = args.instrument

    # update this to use the Instruments enum if needed,
    # for now we can just do a simple match-case
    match instrument_name.upper():
        case "ALFOSC":
            instrument = ALFOSC()
        case "NOTCAM":
            instrument = NOTCAM()

    show_plots = args.show_plots
    if show_plots is None:
        show_plots = False

    pipeline = ReductionPipeline(
        instrument, raw_data_path, output_dir=output_dir, show_plots=show_plots
    )

    if not args.skip_reduction:
        pipeline.run_pipeline()

    if not args.skip_phot_calib:
        if not args.skip_WCS_refinement:
            pipeline.run_photometric_calibrations()
        else:
            pipeline.run_photometric_calibrations(skip_WCS_refinement=True)


if __name__ == "__main__":
    main()
