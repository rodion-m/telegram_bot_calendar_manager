# logging_setup.py
import logging


class LoggerSetup:
    """Sets up the logging configuration."""

    @staticmethod
    def setup_logging(level: str = "DEBUG"):
        logging.basicConfig(
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            level=getattr(logging, level.upper(), logging.DEBUG)
        )
        logger = logging.getLogger(__name__)
        return logger