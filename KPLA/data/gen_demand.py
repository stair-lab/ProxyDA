"""
Data generation code for the Demand dataset
the code is copied from https://github.com/liyuan9988/DeepFeatureProxyVariable/tree/master
Redistribution of the source code under MIT License
Date modified: Sep 19 2023
"""
import numpy as np
from numpy.random import default_rng

from .data_class import dfaDataSet


def psi(t: np.ndarray) -> np.ndarray:
    return 2 * ((t - 5) ** 4 / 600 + np.exp(-4 * (t - 5) ** 2) + t / 10 - 2)

def generate_demand(n_sample: int, rng):
        demand = rng.uniform(0, 10, n_sample)     #U
        return demand


def generatate_demand_core(demand, n_sample: int, rng):
    cost1 = 2 * np.sin(demand * np.pi * 2 / 10) + rng.normal(0, 1.0, n_sample) #x1
    cost2 = 2 * np.cos(demand * np.pi * 2 / 10) + rng.normal(0, 1.0, n_sample) #x2
    price = 35 + (cost1 + 3) * psi(demand) + cost2 + rng.normal(0, 1.0, n_sample) #c
    views = 7 * psi(demand) + 45 + rng.normal(0, 1.0, n_sample) #W
    outcome = cal_outcome(price, views, demand) #Y
    return cost1, cost2, price, views, outcome

def generate_demand_dataset(gen_u, n_sample: int, seed=42, **kwargs):
    rng = default_rng(seed=seed)
    demand = gen_u(n_sample, rng)
    cost1, cost2, price, views, outcome = generatate_demand_core(demand, n_sample, rng)
    outcome = (outcome + rng.normal(0, 1.0, n_sample)).astype(float)
    return dfaDataSet(C=price[:, np.newaxis],
                    X=np.c_[cost1, cost2],
                    W=views[:, np.newaxis],
                    Y=outcome[:, np.newaxis])


def cal_outcome(price, views, demand):
    return np.clip(np.exp((views - price) / 10.0), None, 5.0) * price - 5 * psi(demand)

