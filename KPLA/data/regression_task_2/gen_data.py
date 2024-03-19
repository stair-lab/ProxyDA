"""Generate data for source and target domains."""

# Author: Katherine Tsai <kt14@illinois.edu>, Nicole Chiou <nicchiou@stanford.edu>
# MIT License


from jax import random
import numpy as np
import jax.numpy as jnp

from KPLA.data.regression_task_1.data_generator import (
    gen_U,
    gen_X,
    gen_W,
    gen_Y,
)


def gen_source_data(n, s1, s2, var, mean, seed_list):
    data_list = []
    for env_id, sd in enumerate(seed_list):
        seed1 = sd + 5446
        seed2 = sd + 3569
        seed3 = sd + 10
        seed4 = sd + 1572

        keyu = random.PRNGKey(seed1)
        keyu, *subkeysu = random.split(keyu, 4)

        keyx = random.PRNGKey(seed2)
        keyx, *subkeysx = random.split(keyx, 4)

        keyw = random.PRNGKey(seed3)
        keyw, *subkeysw = random.split(keyw, 4)

        keyz = random.PRNGKey(seed4)
        keyz, *subkeysz = random.split(keyz, 4)

        U = (jnp.asarray(gen_U(env_id, n, s1, s2, key=subkeysu))).T
        X = (jnp.asarray(gen_X(n, var, mean, key=subkeysx))).T
        W = (jnp.asarray(gen_W(U, n, key=subkeysw))).T
        Y = gen_Y(X, U)

        # Encode one-hot
        Z = np.zeros((n, 3))
        Z[:, env_id] = 1
        Z = jnp.asarray(Z)

        data = {}

        data["U"] = jnp.array(U)
        data["X"] = jnp.array(X)
        data["W"] = jnp.array(W)
        data["Z"] = jnp.array(Z)
        data["Y"] = jnp.array(Y)
        data_list.append(data)

    return data_list


def gen_target_data(n_env, n, s1, s2, var, mean, seed_list):
    data_list = []
    for sd in seed_list:
        seed1 = sd + 5446
        seed2 = sd + 3569
        seed3 = sd + 10
        seed4 = sd + 1572

        keyu = random.PRNGKey(seed1)
        keyu, *subkeysu = random.split(keyu, 4)

        keyx = random.PRNGKey(seed2)
        keyx, *subkeysx = random.split(keyx, 4)

        keyw = random.PRNGKey(seed3)
        keyw, *subkeysw = random.split(keyw, 4)

        keyz = random.PRNGKey(seed4)
        keyz, *subkeysz = random.split(keyz, 4)

        U = (jnp.asarray(gen_U(n_env, n, s1, s2, key=subkeysu))).T
        X = (jnp.asarray(gen_X(n, var, mean, key=subkeysx))).T
        W = (jnp.asarray(gen_W(U, n, key=subkeysw))).T
        Y = gen_Y(X, U)

        # Encode one-hot
        Z = np.zeros((n, 3))
        Z[:, -1] = 1
        Z = jnp.asarray(Z)

        data = {}

        data["U"] = jnp.array(U)
        data["X"] = jnp.array(X)
        data["W"] = jnp.array(W)
        data["Z"] = jnp.array(Z)
        data["Y"] = jnp.array(Y)
        data_list.append(data)

    return data_list
