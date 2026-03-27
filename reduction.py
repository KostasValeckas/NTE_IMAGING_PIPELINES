from instruments import Instrument
from logger import init_logger
from sorting import sort_data, create_setup_table, create_bias_table, create_flat_table


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
        self.bad_pixel_masks = None

        self.master_flats = None

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

        self.master_biases, self.bad_pixel_masks = self.instrument.make_master_bias(
            self.raw_data_path,
            self.output_dir,
            self.bias_configurations,
            self.logger,
            show_plots=self.show_plots,
        )

        self.flat_configurations = create_flat_table(
            self.instrument,
            self.logger,
            self.raw_data_path,
            self.output_dir,
            self.setup_table,
            self.flat_files,
        )

        self.master_flats = self.instrument.make_master_flat(
            self.raw_data_path,
            self.output_dir,
            self.flat_configurations,
            self.logger,
            bad_pixel_masks = self.bad_pixel_masks,
            bias_frames = self.master_biases,
            science_to_bias_map= self.science_to_bias_map,
            show_plots=self.show_plots
        )
