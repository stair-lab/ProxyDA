from typing import Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm

from .nn_structure_for_demand  import build_net_for_demand
from .nn_structure_for_demand_partial import build_net_for_demand_partial
from .nn_structure_for_dsprite import build_net_for_dsprite
from .nn_structure_for_dsprite_partial import build_net_for_dsprite_partial
import logging

logger = logging.getLogger()


def build_extractor(data_name: str) -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:

    if data_name == "demand":
        logger.info("build for demand")
        return build_net_for_demand()

    if data_name == "demand_partial":
        logger.info("build for demand partial")
        return build_net_for_demand_partial()

    if data_name == "dsprites":
        logger.info("build for dsprites")
        return build_net_for_dsprite()

    if data_name == "dsprites_partial":
        logger.info("build for dsprites partial")
        return build_net_for_dsprite_partial()
    else:
        raise ValueError(f"data name {data_name} is not valid")
