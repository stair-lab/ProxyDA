"""
network structure for demand dataset
"""

#Author: Katherine Tsai <kt14@illinois.edu>
#License: MIT

from typing import Tuple

from torch import nn


def build_net_for_multi_demand() -> Tuple[
  nn.Module, nn.Module, nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:

  x1_target_net = nn.Sequential(nn.Linear(1, 32),
                                nn.ReLU(),
                                nn.Linear(32, 16),
                                nn.ReLU(), nn.Linear(16, 8))

  x2_net = nn.Sequential(nn.Linear(1, 32),
                        nn.ReLU(),
                        nn.Linear(32, 16),
                        nn.ReLU(), nn.Linear(16, 8))

  x3_net = nn.Sequential(nn.Linear(1, 32),
                        nn.ReLU(),
                        nn.Linear(32, 16),
                        nn.ReLU(), nn.Linear(16, 8))

  w2_net = nn.Sequential(nn.Linear(1, 32),
                        nn.ReLU(),
                        nn.Linear(32, 16),
                        nn.ReLU(), nn.Linear(16, 8))


  e2_net = nn.Sequential(nn.Linear(1, 32),
                        nn.ReLU(),
                        nn.Linear(32, 16),
                        nn.ReLU(),
                        nn.Linear(16, 8))



  return  x1_target_net, x2_net, x3_net, w2_net, e2_net
