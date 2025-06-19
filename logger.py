##
## EPITECH PROJECT, 2025
## Robocar
## File description:
## logger
##

import sys
import time
import logging

def setup_logger(name: str, level=logging.INFO) -> logging.Logger:
    """
    Configure and return a logger with the specified name and level.
    If the logger already has handlers, it will not add new ones to avoid duplication.
    """

    logger = logging.getLogger(name)
    if logger.hasHandlers():
        # Eviter d’ajouter plusieurs handlers si déjà configuré
        return logger

    logger.setLevel(level)

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Handler console (stdout)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.propagate = False

    return logger

class TimeLogger:
    def __init__(self, message: str, logger: logging.Logger):
        """
        Initialize the TimeLogger with a message, logger, and logging level.
        """
        self.message = message
        self.logger = logger

    def __enter__(self):
        """
        Start the timer and log the start message.
        """
        self.start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        duration = time.perf_counter() - self.start
        self.logger.info(f"{self.message} took {duration:.4f} seconds")
