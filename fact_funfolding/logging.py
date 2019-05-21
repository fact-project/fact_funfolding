import logging


def setup_logging():
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        fmt='%(asctime)s|%(levelname)8s|%(name)s|%(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S',
    )
    handler.setFormatter(formatter)
    logging.getLogger().addHandler(handler)
