"""
Generates synthetic data using the generating functions provided in
`data/classification_task/data_lsa.py`.
The output is written to `./tmp_data` by default.
"""

import argparse
import os

from KPLA.data.classification_task.data_generator import (
    generate_data,
    tidy_w,
    pack_to_df,
)

parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=10000)
parser.add_argument("--seed", type=int, default=192)
parser.add_argument("--num_seeds", type=int, default=10)
parser.add_argument("--outdir", type=str, default="./tmp_data")
args = parser.parse_args()

os.makedirs(args.outdir, exist_ok=True)

p_u_range = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
w_coeff_list = [1, 2, 3]
partition_dict = {"train": 0.7, "val": 0.1, "test": 0.2}
for seed in range(args.seed, args.seed + args.num_seeds):
    result = {}
    for i, p_u_0 in enumerate(p_u_range):
        p_u = [p_u_0, 1 - p_u_0]
        samples_dict = generate_data(
            p_u=p_u,
            seed=i * seed + i,
            num_samples=args.num_samples,
            partition_dict=partition_dict,
        )
        for w_coeff in w_coeff_list:
            samples_dict_tidy = tidy_w(samples_dict, w_value=w_coeff)
            samples_df = pack_to_df(samples_dict_tidy)
            filename = (
                "synthetic_multivariate_num_samples_10000_w_coeff_"
                + f"{w_coeff}_p_u_0_{p_u_0}_{seed}.csv"
            )
            samples_df.to_csv(os.path.join(args.outdir, filename), index=False)
