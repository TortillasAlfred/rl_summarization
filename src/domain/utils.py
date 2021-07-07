import logging
import sys
import numpy as np
import torch
import os
import random

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
    return logging.Formatter("%(asctime)s : %(levelname)s : %(message)s")


def get_formatted_desc(desc=""):
    return f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S,%f')[:-3]} : INFO : {desc}"


def datetime_tqdm(iterable, *, desc="", **kwargs):
    return tqdm(iterable, desc=get_formatted_desc(desc), **kwargs)


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def get_name_from_config(config):
    keys = ["encoder", "encoder_size", "ucb_sampling", "rescale_targets", "seed"]

    return "--".join([f"{key}={getattr(config, key)}" for key in keys])


def collect_environment_variables(config):
    for name, val in vars(config).items():
        if isinstance(val, str):
            setattr(config, name, os.path.expandvars(val))


def nansum(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs)
