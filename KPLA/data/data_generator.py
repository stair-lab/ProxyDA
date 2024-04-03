"""Data generator of multi-source classification task."""

# Author: Katherine Tsai <kt14@illinois.edu>
# MIT License

import jax.numpy as jnp
import numpy as np
from KPLA.data.data_lsa import generate_multienv_data, tidy_w


def gen_multienv_class_discrete_z(
    z_env,
    seed,
    num_samples,
    task=None,
    partition_dict=dict({"train": 0.8, "test": 0.2}),
):
    """
    Args:
      z_inv: environment id, int
      seed: random seed, int
      num_samples: number of samples, int
      task: multisource classification task source/target distributions, int
      partiation_dict: partition, dict
    """

    samples_dict = generate_multienv_data(
        z_indicator=z_env,
        seed=seed,
        num_samples=num_samples,
        task=task,
        partition_dict=partition_dict,
    )

    samples_dict_tidy = tidy_w(samples_dict, w_value=3)
    keys = samples_dict_tidy["train"].keys()

    source_train = {}
    source_test = {}

    maps = {
        "x": "X",
        "w": "W",
        "c": "C",
        "u": "U",
        "y_one_hot": "Y_one_hot",
        "y": "Y",
    }
    for key in keys:
        if key in maps.keys():
            source_train[maps[key]] = jnp.array(
                samples_dict_tidy["train"][key]
            )
            source_test[maps[key]] = jnp.array(samples_dict_tidy["test"][key])

    source_train["Z"] = jnp.ones(source_train["X"].shape[0]) * z_env
    source_test["Z"] = jnp.ones(source_test["X"].shape[0]) * z_env

    z = np.zeros(
        (source_train["X"].shape[0], samples_dict_tidy["train"]["n_env"])
    )
    z[:, z_env] = 1.0
    source_train["Z_one_hot"] = jnp.array(z)

    z = np.zeros(
        (source_test["X"].shape[0], samples_dict_tidy["test"]["n_env"])
    )
    z[:, z_env] = 1.0
    source_test["Z_one_hot"] = jnp.array(z)

    return source_train, source_test


def convert_to_numpy(data_dict):
    numpy_dict = {}
    for k, v in data_dict.items():
        numpy_dict[k] = np.asarray(v)
    return numpy_dict
