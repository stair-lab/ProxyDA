"""
Multi-source adaptation with baselines and sweep over target domains.
Using simulated regression task 2.
"""

# Author: Katherine Tsai <kt14@illinois.edu>, Nicole Chiou <nicchiou@stanford.edu>
# MIT License

import argparse
import copy
import os
import itertools
import numpy as np
import pandas as pd

from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


from KPLA.data.regression_task_2.gen_data import (
    gen_source_data,
    gen_target_data,
)

from KPLA.baselines.model_select import select_kernel_ridge_model
from KPLA.baselines.multi_source_ccm import (
    MultiSouceSimpleAdapt,
    MultiSourceUniformReg,
)
from KPLA.baselines.multi_source_cat import MultiSourceCatReg


parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=2000)
parser.add_argument("--n_env", type=int, default=2)
parser.add_argument("--var", type=float, default=1.0)
parser.add_argument("--mean", type=float, default=0)
parser.add_argument("--kernel", type=str, default="rbf")
parser.add_argument("--add_w", type=bool, default=False)
parser.add_argument("--outdir", type=str, default="./")
parser.add_argument("--verbose", type=bool, default=False)
args = parser.parse_args()

out_dir = args.outdir
os.makedirs(out_dir, exist_ok=True)


a_list = [1, 2, 3, 4, 5]
b_list = [1, 2, 3, 4, 5]

for sdj in range(10):
    main_summary = pd.DataFrame()

    for a, b in itertools.product(a_list, b_list):

        ####################
        # Generate data    #
        ####################

        seed_list = {}
        sd_lst = [
            5949,
            7422,
            4388,
            2807,
            5654,
            5518,
            1816,
            1102,
            9886,
            1656,
            4379,
            2029,
            8455,
            4987,
            4259,
            2533,
            9783,
            7987,
            1009,
            2297,
        ]

        # Generate data from source domain
        sd_train_list = sd_lst[sdj * args.n_env : args.n_env * (sdj + 1)]
        source_train = gen_source_data(
            args.n, a, b, args.var, args.mean, sd_train_list, reshape_X=True
        )
        source_cat_train = {}
        for k in ["U", "X", "W", "Z", "Y"]:
            temp = np.asarray(copy.deepcopy(source_train[0][k]))
            for data in source_train[1::]:
                temp = np.concatenate((temp, np.asarray(data[k])))
            source_cat_train[k] = temp

        # Test set only has 1000 samples
        sd_test_list = sd_lst[(9 - sdj) * args.n_env : args.n_env * (10 - sdj)]
        source_test = gen_source_data(
            1000, a, b, args.var, args.mean, sd_test_list, reshape_X=True
        )
        source_cat_test = {}
        for k in ["U", "X", "W", "Z", "Y"]:
            temp = np.asarray(copy.deepcopy(source_test[0][k]))
            for data in source_test[1::]:
                temp = np.concatenate((temp, np.asarray(data[k])))
            source_cat_test[k] = temp

        # Generate data from target domain
        target_train = gen_target_data(
            args.n_env,
            args.n * 2,
            a,
            b,
            args.var,
            args.mean,
            [sd_lst[sdj]],
            reshape_X=True,
        )
        for k in ["U", "X", "W", "Z", "Y"]:
            target_train[0][k] = np.asarray(target_train[0][k])
        target_test = gen_target_data(
            args.n_env,
            1000,
            a,
            b,
            args.var,
            args.mean,
            [sd_lst[sdj]],
            reshape_X=True,
        )
        for k in ["U", "X", "W", "Z", "Y"]:
            target_test[0][k] = np.asarray(target_test[0][k])

        if args.verbose:
            print("Data generation complete")
            print("Number of source environments:", len(source_train))
            print(
                "Source_train number of samples: ",
                source_train[0]["X"].shape[0] * args.n_env,
            )
            print(
                "Source_test  number of samples: ",
                source_test[0]["X"].shape[0],
            )
            print("Number of target environments:", len(target_train))
            print(
                "Target_train number of samples: ",
                target_train[0]["X"].shape[0],
            )
            print(
                "Target_test  number of samples: ",
                target_test[0]["X"].shape[0],
            )

        RESULTS = {}
        metrics = []
        MODEL = KernelRidge
        # Regress with W
        if args.add_w:
            source_cat_train["covar"] = np.stack(
                (source_cat_train["X"].squeeze(), source_cat_train["W"])
            ).T
            source_cat_test["covar"] = np.stack(
                (source_cat_test["X"].squeeze(), source_cat_test["W"])
            ).T
            target_train[0]["covar"] = np.stack(
                (target_train[0]["X"].squeeze(), target_train[0]["W"])
            ).T
            target_test[0]["covar"] = np.stack(
                (target_test[0]["X"].squeeze(), target_test[0]["W"])
            ).T

        else:
            source_cat_train["covar"] = source_cat_train["X"].reshape(-1, 1)
            source_cat_test["covar"] = source_cat_test["X"].reshape(-1, 1)
            target_train[0]["covar"] = target_train[0]["X"].reshape(-1, 1)
            target_test[0]["covar"] = target_test[0]["X"].reshape(-1, 1)

        N_PARAMS = 10
        N_FOLD = 5

        ####################
        # Source ERM       #
        ####################
        kr_model = MODEL(kernel=args.kernel)
        source_erm, source_erm_hparams = select_kernel_ridge_model(
            kr_model,
            source_cat_train["covar"],
            source_cat_train["Y"],
            n_params=N_PARAMS,
            n_fold=N_FOLD,
            min_val=-4,
            max_val=3,
        )
        erm_metrics = {"approach": "ERM"}
        erm_metrics["source -> source"] = mean_squared_error(
            source_cat_test["Y"], source_erm.predict(source_cat_test["covar"])
        )
        erm_metrics["source -> target"] = mean_squared_error(
            target_test[0]["Y"], source_erm.predict(target_test[0]["covar"])
        )

        ####################
        # Target ERM       #
        ####################
        kr_model = MODEL(kernel=args.kernel)
        target_erm, target_erm_hparams = select_kernel_ridge_model(
            kr_model,
            target_train[0]["covar"],
            target_train[0]["Y"],
            n_params=N_PARAMS,
            n_fold=N_FOLD,
            min_val=-4,
            max_val=3,
        )
        erm_metrics["target -> target"] = mean_squared_error(
            target_test[0]["Y"], target_erm.predict(target_test[0]["covar"])
        )
        erm_metrics["target -> source"] = mean_squared_error(
            source_cat_test["Y"], target_erm.predict(source_cat_test["covar"])
        )

        RESULTS["target_erm"] = target_erm

        metrics.append(erm_metrics)
        if args.verbose:
            print("ERM metrics")
            print(erm_metrics)

        #######################
        # Multisource uniform #
        #######################
        msur_metrics = {"approach": "multisource uniform"}

        source_msur = MultiSourceUniformReg(n_env=args.n_env, max_iter=3000)

        source_msur.fit(source_train)
        msur_metrics["source -> source"] = mean_squared_error(
            source_cat_test["Y"],
            source_msur.predict(source_cat_test["covar"]),
        )
        msur_metrics["source -> target"] = mean_squared_error(
            target_test[0]["Y"],
            source_msur.predict(target_test[0]["covar"]),
        )

        metrics.append(msur_metrics)
        if args.verbose:
            print("MSUR metrics")
            print(msur_metrics)

        ####################
        # Multisource cat. #
        ####################
        msca_metrics = {"approach": "multisource cat"}

        source_msca = MultiSourceCatReg(max_iter=3000)
        source_msca.fit(source_train)

        msca_metrics["source -> source"] = mean_squared_error(
            source_cat_test["Y"].reshape(-1, 1),
            source_msca.predict(source_cat_test["covar"]),
        )
        msca_metrics["source -> target"] = mean_squared_error(
            target_test[0]["Y"].reshape(-1, 1),
            source_msca.predict(target_test[0]["covar"]),
        )

        metrics.append(msca_metrics)
        if args.verbose:
            print("MSCA metrics")
            print(msca_metrics)

        #############################
        # Multisource simple adapt. #
        #############################
        mssa_metrics = {"approach": "multisource simple adapt"}

        def run_single_loop(params, n_folds):
            kf = KFold(n_splits=n_folds, random_state=None, shuffle=False)
            errs = []
            best_err_i = np.inf
            best_model_i = None
            split_idx = kf.split(source_train[0]["X"])

            for _, (train_idx, test_idx) in enumerate(split_idx):

                def split_data_cv(sidx, t_list):
                    list_len = len(t_list)
                    return [
                        {k: v[sidx] for k, v in t_list[j].items()}
                        for j in range(list_len)
                    ]

                source_train_cv_train = split_data_cv(train_idx, source_train)
                source_train_cv_test = split_data_cv(test_idx, source_train)

                mssa = MultiSouceSimpleAdapt(
                    n_env=len(source_train_cv_train),
                    bandwidth=params,
                    max_iter=3000,
                    task="r",
                )

                mssa.fit(source_train_cv_train)
                # Select parameters from source
                acc_err = 0
                for sd in source_train_cv_test:
                    acc_err += mean_squared_error(
                        sd["Y"].ravel(), mssa.predict(sd["X"])
                    )
                errs.append(acc_err)
                # Select parameters from target
                if acc_err < best_err_i:
                    best_err_i = acc_err
                    best_model_i = mssa
                    b_params = {"bandwidth": params}

            return best_model_i, np.mean(errs), b_params

        n_params = 20
        n_fold = 5
        min_val = -4
        max_val = 2

        length_scale = np.logspace(min_val, max_val, n_params)

        params_batch = length_scale

        results = [0] * length_scale.shape[0]
        for i in range(length_scale.shape[0]):
            results[i] = run_single_loop(params_batch[i], n_fold)

        model_batch = [r[0] for r in results]
        err_batch = [r[1] for r in results]
        params_batch = [r[2] for r in results]
        idx = np.argmin(err_batch)

        best_msa = model_batch[idx]
        best_params = params_batch[idx]

        if args.verbose:
            print("optimal parameters", params_batch[idx])

        best_msa.fit(source_train)

        source_msa = MultiSouceSimpleAdapt(
            n_env=args.n_env,
            bandwidth=best_params["bandwidth"],
            max_iter=3000,
            task="r",
        )
        source_msa.fit(source_train)
        mssa_metrics["source -> source"] = mean_squared_error(
            source_cat_test["Y"], source_msa.predict(source_cat_test["X"])
        )
        mssa_metrics["source -> target"] = mean_squared_error(
            target_test[0]["Y"], source_msa.predict(target_test[0]["X"])
        )

        metrics.append(mssa_metrics)
        if args.verbose:
            print("MMSA metrics")
            print(mssa_metrics)

        summary = pd.DataFrame.from_records(metrics)
        summary["a"] = a
        summary["b"] = b
        summary["seed"] = sdj
        main_summary = pd.concat([main_summary, summary])

    main_summary.to_csv(
        os.path.join(out_dir, f"sweep_baseline_seed_{sdj}.csv"), index=False
    )
