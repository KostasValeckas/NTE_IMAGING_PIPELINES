import logging
from colorama import Fore, Style, init
import pathlib

"""
A logging module that get initialized during every reduction run. A log 
file is created and stored in the directory of the raw data, with the name 
of the raw_data_directory_pipeline.log Will be overrriden on repeated executions.
"""


def init_logger(path):

    # Initialize colorama
    init(autoreset=True)

    # Create a custom logger

    logger = logging.getLogger("NTE_IMAGING_PIPELINE")

    # Close and remove leftover handlers
    for handler in logger.handlers:
        handler.close()
        logger.removeHandler(handler)

    # Configure logging level
    logger.setLevel(logging.INFO)

    config_file_dir = pathlib.Path(path)
    config_file_name = pathlib.Path(path).stem + "_pipeline.log"

    log_file_path = config_file_dir / config_file_name

    # we add different information when we print to console and when we write to
    # file, so we need two handlers
    fh = logging.FileHandler(log_file_path)
    ch = logging.StreamHandler()

    # Create a custom formatter - this allows different colors for different log levels
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            if record.levelno == logging.INFO:
                record.levelname = f"{Fore.GREEN}{record.levelname}{Style.RESET_ALL}"
            elif record.levelno == logging.WARNING:
                record.levelname = f"{Fore.YELLOW}{record.levelname}{Style.RESET_ALL}"
            elif record.levelno == logging.ERROR:
                record.levelname = f"{Fore.RED}{record.levelname}{Style.RESET_ALL}"
            elif record.levelno == logging.CRITICAL:
                record.levelname = (
                    f"{Fore.RED}{record.levelname}{Style.BRIGHT}{Style.RESET_ALL}"
                )
            return super().format(record)

    # For filelogging we add the dates, for console logging we don't
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    color_formatter = CustomFormatter(
        "%(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )

    fh.setFormatter(formatter)
    ch.setFormatter(color_formatter)

    # Add both handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Log messages
    logger.info("Logger initialized. Log will be saved in " + fh.baseFilename + ".")

    return logger
