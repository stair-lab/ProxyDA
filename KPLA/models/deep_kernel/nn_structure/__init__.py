"""
network loader
"""

#Author: Katherine Tsai <kt14@illinois.edu>
#License: MIT
from typing import Tuple

from torch import nn


from .nn_structure_for_demand  import build_net_for_demand
from .nn_structure_for_multi_demand  import build_net_for_multi_demand
from .nn_structure_for_dsprite import build_net_for_dsprite
import logging

logger = logging.getLogger()


def build_extractor(data_name: str) -> Tuple[
  nn.Module, nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:

  if data_name == "demand":
    logger.info("build for demand")
    return build_net_for_demand()

  if data_name == "multi_demand":
    logger.info("build for multi demand")
    return build_net_for_multi_demand()

  if data_name == "dsprites":
    logger.info("build for dsprites")
    return build_net_for_dsprite()


  else:
    raise ValueError(f"data name {data_name} is not valid")
