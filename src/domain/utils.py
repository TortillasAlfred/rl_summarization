import logging
import sys
import time

from datetime import datetime
from tqdm import tqdm


def configure_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = default_formatter()

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)


def default_formatter():
    return logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')


def get_formatted_desc(desc=''):
    return f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} : INFO : {desc}"


def datetime_tqdm(iterable, *, desc='', **kwargs):
    return tqdm(iterable, desc=get_formatted_desc(desc), **kwargs)
