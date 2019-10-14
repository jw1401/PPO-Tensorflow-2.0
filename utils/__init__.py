
import logging
from .logger import log, Logger, log2logger
from .misc import singleton


"""
    # Set Logging 
    -------------
"""


class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[0;37;40m"
    magenta = "\x1b[1;35;21m"
    blue = "\x1b[0;36;40m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(name)s|%(levelname)s: %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: blue + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class CustomFormatter_mlagents(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    grey = "\x1b[0;37;40m"
    magenta = "\x1b[1;35;21m"
    blue = "\x1b[0;36;40m"
    yellow = "\x1b[33;21m"
    red = "\x1b[1;31;40m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "\n%(name)s - %(levelname)s:\n---------------------\n" + \
        grey + "%(message)s \n"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: yellow + format + reset,
        logging.WARNING: magenta + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def set_loggers():

    # create logger with 'spam_application'
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(CustomFormatter())
    logger.handlers = []
    if not logger.handlers:
        logger.addHandler(ch)

    ml_logger = logging.getLogger("mlagents.envs")
    ml_logger.disabled = False
    mlch = logging.StreamHandler()
    mlch.setLevel(logging.INFO)
    mlch.setFormatter(CustomFormatter_mlagents())
    ml_logger.propagate = False
    ml_logger.handlers = []
    if not ml_logger.handlers:
        ml_logger.addHandler(mlch)

    logging.getLogger("tensorflow").disabled = True


set_loggers()
