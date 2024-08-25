# Copyright (c) 2023-present, SUSTech-ML.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


import random
import math
import warnings

import numpy as np
import torch

__all__ = ["fix_randomness", "DimensionError", "get_device"]


def fix_randomness(seed=0):
    """
    Fix the random seed for python, torch, numpy.

    :param seed: the random seed
    """
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


class DimensionError(Exception):
    pass


def get_device(model):
    """
    Get the device of Torch model.

    :param model: a Pytorch model. If None, it uses GPU when the cuda is available, otherwise it uses CPU。

    :return: the device in use
    """
    if model is None:
        if not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            cuda_idx = torch.cuda.current_device()
            device = torch.device(f"cuda:{cuda_idx}")
    else:
        device = next(model.parameters()).device
    return device


def calculate_conformal_value(scores, alpha):
    """
    Calculate the 1-alpha quantile of scores.
    
    :param scores: non-conformity scores.
    :param alpha: a significance level.
    
    :return: the threshold which is use to construct prediction sets.
    """
    if alpha >= 1 or alpha <= 0:
            raise ValueError("Significance level 'alpha' must be in (0,1).")
    if len(scores) == 0:
        warnings.warn(
            "The number of scores is 0, which is a invalid scores. To avoid program crash, the threshold is set as torch.inf.")
        return torch.inf
    qunatile_value = math.ceil(scores.shape[0] + 1) * (1 - alpha) / scores.shape[0]

    if qunatile_value > 1:
        warnings.warn(
            "The value of quantile exceeds 1. It should be a value in (0,1). To avoid program crash, the threshold is set as torch.inf.")
        return torch.inf

    return torch.quantile(scores, qunatile_value, dim=0).to(scores.device)