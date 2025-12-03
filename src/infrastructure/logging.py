import logging
import sys

def setup_logging():
    """
    Configure the application's root logger.
    
    Sets the log level to INFO, uses the format "timestamp - logger name - level - message" for records, and attaches a StreamHandler that writes logs to stdout.
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )