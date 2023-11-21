from typing import Optional, Tuple
import torch
from torch import nn


def build_net_for_demand_partial() -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:
    x1_source_net = nn.Sequential(nn.Linear(2, 32),
                                        nn.ReLU(),
                                        nn.Linear(32, 16),
                                        nn.ReLU(), nn.Linear(16, 8))

    x1_target_net = nn.Sequential(nn.Linear(2, 32),
                                        nn.ReLU(),
                                        nn.Linear(32, 16),
                                        nn.ReLU(), nn.Linear(16, 8))

    x2_net = nn.Sequential(nn.Linear(2, 32),
                                        nn.ReLU(),
                                        nn.Linear(32, 16),
                                        nn.ReLU(), nn.Linear(16, 8))

    x4_net = nn.Sequential(nn.Linear(2, 32),
                                        nn.ReLU(),
                                        nn.Linear(32, 16),
                                        nn.ReLU(), nn.Linear(16, 8))
    w2_net = nn.Sequential(nn.Linear(1, 32),
                                      nn.ReLU(),
                                      nn.Linear(32, 16),
                                      nn.ReLU(), nn.Linear(16, 8))

    
    c2_net = nn.Sequential(nn.Linear(1, 32),
                                      nn.ReLU(),
                                      nn.Linear(32, 16),
                                      nn.ReLU(),
                                      nn.Linear(16, 8))

    c3_net = nn.Sequential(nn.Linear(1, 32),
                                      nn.ReLU(),
                                      nn.Linear(32, 16),
                                      nn.ReLU(),
                                      nn.Linear(16, 8))



    return x1_source_net, x1_target_net, x2_net, x4_net, w2_net, c2_net, c3_net