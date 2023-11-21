from typing import Optional, Tuple
import torch
from torch import nn
from torch.nn.utils import spectral_norm
from torch.nn import functional as F


def build_net_for_dsprite_partial() -> Tuple[
    nn.Module, nn.Module, nn.Module, nn.Module, nn.Module, nn.Module, nn.Module]:
    x1_source_net = nn.Sequential(spectral_norm(nn.Linear(64 * 64, 1024)),
                                      nn.ReLU(),
                                      spectral_norm(nn.Linear(1024, 512)),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(512),
                                      spectral_norm(nn.Linear(512, 512)),
                                      nn.ReLU(),
                                      spectral_norm(nn.Linear(512, 32)),
                                      nn.ReLU())

    x1_target_net = nn.Sequential(spectral_norm(nn.Linear(64 * 64, 1024)),
                                      nn.ReLU(),
                                      spectral_norm(nn.Linear(1024, 512)),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(512),
                                      spectral_norm(nn.Linear(512, 512)),
                                      nn.ReLU(),
                                      spectral_norm(nn.Linear(512, 32)),
                                      nn.ReLU())

    x2_net = nn.Sequential(spectral_norm(nn.Linear(64 * 64, 1024)),
                                      nn.ReLU(),
                                      spectral_norm(nn.Linear(1024, 512)),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(512),
                                      spectral_norm(nn.Linear(512, 512)),
                                      nn.ReLU(),
                                      spectral_norm(nn.Linear(512, 32)),
                                      nn.ReLU())

    x4_net = nn.Sequential(spectral_norm(nn.Linear(64 * 64, 1024)),
                                      nn.ReLU(),
                                      spectral_norm(nn.Linear(1024, 512)),
                                      nn.ReLU(),
                                      nn.BatchNorm1d(512),
                                      spectral_norm(nn.Linear(512, 512)),
                                      nn.ReLU(),
                                      spectral_norm(nn.Linear(512, 32)),
                                      nn.ReLU())

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