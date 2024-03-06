"""
Data generator for the multi-source regression task.
Follows the data generation process (D.3).
"""

# Author: Katherine Tsai <kt14@illinois.edu>, Nicole Chiou <nicchiou@stanford.edu>
# MIT License

from jax import random
import numpy as np
import jax.numpy as jnp


def gen_U(Z, n, s1, s2, key):
    """Sample U."""
    if Z == 0:
        # Reversed setting (0.1, 0.9), original setting (0.9, 0.1)
        print("Z is 0")
        U = random.choice(key[0], jnp.arange(2), (n,), p=np.array([0.9, 0.1]))
    elif Z == 1:
        print("Z is 1")
        U = random.choice(key[0], jnp.arange(2), (n,), p=np.array([0.1, 0.9]))
    else:  # Target domain
        print("Z is 2")
        U = random.choice(key[0], jnp.arange(2), (n,), p=np.array([s1, s2]))
    return U


def gen_X(n, var, mean, key):
    """Sample X."""
    X = random.normal(key[0], (n,)) * var + mean
    return X


def gen_W(U, n, key):
    """Sample W|U."""
    W1 = random.normal(key[0], (n,)) * 0.01 - 1
    W2 = random.normal(key[0], (n,)) * 0.01 + 1
    W = (1 - U) * W1 + U * W2
    return W


def gen_Y(X, U):
    """Sample Y|X,U."""
    Y1 = -X
    Y2 = X
    Y = (1 - U) * Y1 + U * Y2
    return Y
