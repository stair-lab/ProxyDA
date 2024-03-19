# coding: utf-8
# Copyright 2022 Google LLC.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Generates synthetic data using the generating functions provided in
`data/data_lsa.py`. The output is written to `./tmp_data` by default.
"""

import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

from KPLA.data.data_lsa import MultiWSimulator


def get_squeezed_df(data_dict: dict) -> pd.DataFrame:
    """
    Converts a dict of numpy arrays into a DataFrame, extracting columns of
    arrays into separate DataFrame columns.
    """
    temp = {}
    for key, value in data_dict.items():
        squeezed_array = np.squeeze(value)
        if len(squeezed_array.shape) == 1:
            temp[key] = squeezed_array
        elif len(squeezed_array.shape) > 1:
            for i in range(value.shape[1]):
                temp[f"{key}_{i}"] = np.squeeze(value[:, i])
    return pd.DataFrame(temp)


def process_data(data_dict, w_cols=["w_1", "w_2", "w_3"]):
    result = data_dict.copy()
    for w_col in w_cols:
        result[f"{w_col}_binary"] = 1.0 * (result[f"{w_col}"] > 0)
        result[f"{w_col}_one_hot"] = OneHotEncoder(
            sparse_output=False
        ).fit_transform(result[f"{w_col}_binary"])
    result["u_one_hot"] = OneHotEncoder(sparse_output=False).fit_transform(
        result["u"].reshape(-1, 1)
    )
    return result


def generate_data(p_u, seed, num_samples, partition_dict, param_dict=None):
    sim = MultiWSimulator(param_dict=param_dict)
    samples_dict = {}
    for i, (partition_key, partition_frac) in enumerate(
        partition_dict.items()
    ):
        num_samples_partition = int(partition_frac * num_samples)
        sim.update_param_dict(num_samples=num_samples_partition, p_u=p_u)
        samples_dict[partition_key] = process_data(
            sim.get_samples(seed=seed + 15 * i)
        )
    return samples_dict


def tidy_w(data_dict, w_value):
    result = data_dict.copy()
    for key in result.keys():
        result[key]["w"] = result[key][f"w_{w_value}"]
        result[key]["w_binary"] = result[key][f"w_{w_value}_binary"]
        result[key]["w_one_hot"] = result[key][f"w_{w_value}_one_hot"]
    return result


def pack_to_df(samples_dict):
    """Convert data to DataFrame format."""
    return (
        pd.concat(
            {
                key: get_squeezed_df(value)
                for key, value in samples_dict.items()
            }
        )
        .reset_index(level=-1, drop=True)
        .rename_axis("partition")
        .reset_index()
    )


def extract_from_df(
    samples_df,
    cols=[
        "u",
        "x",
        "w",
        "c",
        "c_logits",
        "y",
        "y_logits",
        "y_one_hot",
        "u_one_hot",
        "x_scaled",
        "w_1",
        "w_1_binary",
        "w_1_one_hot",
        "w_2",
        "w_2_binary",
        "w_2_one_hot",
        "w_2_binary",
        "w_2_one_hot",
    ],
):
    """Extracts dict of numpy arrays from DataFrame."""
    result = {}
    for col in cols:
        if col in samples_df.columns:
            result[col] = samples_df[col].values
        else:
            match_str = f"^{col}_\d$"
            r = re.compile(match_str, re.IGNORECASE)
            matching_columns = list(filter(r.match, samples_df.columns))
            if len(matching_columns) == 0:
                continue
            result[col] = samples_df[matching_columns].to_numpy()
    return result


def extract_from_df_nested(
    samples_df,
    cols=[
        "u",
        "x",
        "w",
        "c",
        "c_logits",
        "y",
        "y_logits",
        "y_one_hot",
        "w_binary",
        "w_one_hot",
        "u_one_hot",
        "x_scaled",
    ],
):
    """
    Extracts nested dict of numpy arrays from DataFrame with structure
    {domain: {partition: data}}.
    """
    result = {}
    for domain in samples_df["domain"].unique():
        result[domain] = {}
        domain_df = samples_df.query("domain == @domain")
        for partition in domain_df["partition"].unique():
            partition_df = domain_df.query("partition == @partition")
            result[domain][partition] = extract_from_df(
                partition_df, cols=cols
            )
    return result
