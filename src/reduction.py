from instruments import Instrument
from logger import init_logger
from sorting import sort_data, create_setup_table, create_bias_table, create_flat_table
from photometric_calibs import *
import json


class ReductionPipeline:
    def __init__(
        self, instrument: Instrument, raw_data_path, output_dir=None, show_plots=False
    ):

        self.logger = (
            init_logger(raw_data_path)
            if output_dir is None
            else init_logger(output_dir)
        )

        self.show_plots = show_plots

        if output_dir is not None:
            self.output_dir = output_dir
        else:
            self.output_dir = raw_data_path / "reduced"
            self.output_dir.mkdir(parents=True, exist_ok=True)

        self.instrument = instrument
        self.raw_data_path = raw_data_path

        # to be initialized later
        self.bias_files = None
        self.dark_files = None
        self.flat_files = None
        self.science_files = None

        self.setup_table = None
        self.bias_configurations = None
        self.science_to_bias_map = None
        self.flat_configurations = None
        self.science_to_flat_map = None

        self.master_biases = None
        self.bad_pixel_masks_bias = None
        self.bad_pixel_masks_science = None

        self.master_flats = None

        self.object_setup = None

    def run_pipeline(self):

        self.logger.info("Starting reduction pipeline")

        self.logger.info(f"Output and log files will be saved in: {self.output_dir}...")

        self.logger.info("Sorting data into bias, dark, flat, and science frames")

        self.bias_files, self.dark_files, self.flat_files, self.science_files = (
            sort_data(self.instrument, self.logger, self.raw_data_path, self.output_dir)
        )

        self.logger.info("Determinning setup configurations...")

        self.setup_table = create_setup_table(
            self.instrument,
            self.logger,
            self.raw_data_path,
            self.output_dir,
            self.science_files,
        )

        self.bias_configurations, self.science_to_bias_map = create_bias_table(
            self.instrument,
            self.logger,
            self.raw_data_path,
            self.output_dir,
            self.setup_table,
            self.bias_files,
        )

        self.master_biases, self.bad_pixel_masks_bias = (
            self.instrument.make_master_bias(
                self.raw_data_path,
                self.output_dir,
                self.bias_configurations,
                self.logger,
                show_plots=self.show_plots,
            )
        )

        self.flat_configurations = create_flat_table(
            self.instrument,
            self.logger,
            self.raw_data_path,
            self.output_dir,
            self.setup_table,
            self.flat_files,
        )

        self.master_flats, self.bad_pixel_masks_science = (
            self.instrument.make_master_flat(
                self.raw_data_path,
                self.output_dir,
                self.flat_configurations,
                self.logger,
                bad_pixel_masks=self.bad_pixel_masks_bias,
                bias_frames=self.master_biases,
                science_to_bias_map=self.science_to_bias_map,
                show_plots=self.show_plots,
            )
        )

        self.object_setup = self.instrument.reduce_science_frames(
            self.raw_data_path,
            self.output_dir,
            self.setup_table,
            self.logger,
            science_to_bias_map=self.science_to_bias_map,
            show_plots=self.show_plots,
        )

        self.object_setup = self.instrument.subtract_sky(
            self.output_dir, self.logger, show_plots=self.show_plots
        )

        # write the object setup to disk for use in photometric calibrations
        object_setup_path = os.path.join(self.output_dir, "object_setup.json")
        with open(object_setup_path, "w") as f:
            json.dump(self.object_setup, f, indent=4)

    def run_photometric_calibrations(self, skip_WCS_refinement=False):

        # load the object setup from disk if not provided
        if self.object_setup is None:
            object_setup_path = os.path.join(self.output_dir, "object_setup.json")
            try:
                with open(object_setup_path, "r") as f:
                    self.object_setup = json.load(f)
            except FileNotFoundError:
                self.logger.error(
                    f"No object setup found at {object_setup_path}. Cannot perform sky subtraction without object setup."
                )
                self.logger.error("Run the reduction first")
                return

        if self.instrument.name == "ALFOSC":
            self.phot_parser = ALFOSC_parser(
                self.output_dir,
                self.logger,
                self.object_setup,
                self.instrument,
                show_plots=self.show_plots,
            )

        if self.instrument.name == "NOTCAM":
            self.phot_parser = NOTCAM_parser(
                self.output_dir,
                self.logger,
                self.object_setup,
                self.instrument,
                show_plots=self.show_plots,
            )

        self.phot_parser.run(skip_WCS_refinement=skip_WCS_refinement)
