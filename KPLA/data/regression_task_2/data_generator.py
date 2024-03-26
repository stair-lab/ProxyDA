"""
Data generator for the multi-source regression task 2.
"""

# Author: Katherine Tsai <kt14@illinois.edu>, Nicole Chiou <nicchiou@stanford.edu>
# MIT License

from jax import random


def gen_U(Z, n, a, b, key):
    """Sample U."""
    if Z == 0:
        # Reversed setting (0.1, 0.9), original setting (0.9, 0.1)
        U = random.beta(key[0], 2, 4, (n,))
    elif Z == 1:
        U = random.beta(key[0], 4, 2, (n,))
    else:  # Target domain
        U = random.beta(key[0], a, b, (n,))
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
    Y = (2 * U - 1) * X
    return Y
