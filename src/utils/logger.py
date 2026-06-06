import logging
import os


def get_logger():

    os.makedirs("logs", exist_ok=True)

    logger = logging.getLogger("churn_project")

    if not logger.handlers:

        logger.setLevel(logging.INFO)

        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s"
        )

        file_handler = logging.FileHandler(
            "logs/project.log"
        )

        file_handler.setFormatter(
            formatter
        )

        stream_handler = logging.StreamHandler()

        stream_handler.setFormatter(
            formatter
        )

        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    return logger