"""Implementation of the base kernel estimator."""

# Author: Katherine Tsai <kt14@illinois.edu>
# MIT License

import pandas as pd

import jax.numpy as jnp
import numpy as np
from KPLA.models.plain_kernel.method import split_data_widx
from KPLA.models.plain_kernel.cme import ConditionalMeanEmbed
from KPLA.models.plain_kernel.method import KernelMethod
from KPLA.models.plain_kernel.kernel_utils import flatten


def concatenate_data(new_data, prev_data):
    """
    new_data: data from new environment
    prev_data: dictionary of data
    """
    keys = prev_data.keys()
    concate_data = {}
    for key in keys:
        if prev_data[key] is None:
            concate_data[key] = new_data[key]
        else:
            concate_data[key] = jnp.concatenate(
                (prev_data[key], new_data[key])
            )

    return concate_data


class MultiKernelMethod(KernelMethod):
    """
    Base estimator for the adaptation
    split_data(), predict(), evaluation(), are implemented by the child class
    """

    def __init__(
        self,
        source_train,
        target_train,
        source_test,
        target_test,
        split,
        scale=1,
        lam_set=None,
        method_set=None,
        kernel_dict=None,
        verbose=False,
    ):
        """Initialize parameters.

        Args:
            source_train: dict, keys: C,W,X,Y
            target_train: dict, keys: C, W, X, Y
            source_test:  dict, keys: X, Y
            target_test:  dict, keys: X, Y
            split: Boolean, split the training dataset or not.
                If True, the samples are evenly split into groups.
                Hence, each estimator receive smaller number of training samples.
            scale: length-scale of the kernel function, default: 1.
            lam_set: a dictionary of tuning parameter,
                set None for leave-one-out estimation
            For example, lam_set={'cme': lam1, 'h0': lam2, 'm0': lam3}
            method_set: a dictionary of optimization methods for
                different estimators, default is 'original'
            kernel_dict: a dictionary of specified kernel functions
            verbose: verbosity flag, bool
        """
        self.source_train = source_train
        self.target_train = target_train
        self.source_test = source_test
        self.target_test = target_test
        self.sc = scale
        self.split = split
        self.fitted = False
        self.verbose = verbose

        if lam_set is None:
            lam_set = {"cme": None, "m0": None}

        self.lam_set = lam_set

        if method_set is None:
            method_set = {"cme": "original", "m0": "original"}
        self.method_set = method_set

        if kernel_dict is None:
            kernel_dict["cme_w_xz"] = {
                "X": "rbf",
                "Z": "rbf",
                "Y": "rbf",
            }  # Y is W
            kernel_dict["cme_w_x"] = {"X": "rbf", "Y": "rbf"}  # Y is W
            kernel_dict["m0"] = {"X": "rbf"}

        self.kernel_dict = kernel_dict

    def fit(self, task="r", train_target=True):
        if self.split:
            self.split_data()
        else:

            n_env = len(self.source_train)
            keys = self.source_train[0].keys()

            cat_source_train = dict(zip(keys, [None] * len(keys)))

            for i in range(n_env):
                cat_source_train = concatenate_data(
                    self.source_train[i], cat_source_train
                )
            self.source_train = [cat_source_train, self.source_train]
            # First train cme(w|c,x), h0 with data pulled across environments,
            # then the second item train the cme(w|x)
            # specifically for each environment
            self.target_train = [self.target_train[0], self.target_train]

        # Learn estimators from the source domain
        if self.verbose:
            print("Fit source domains")
        self.source_estimator = self._fit_one_domain(self.source_train, task)

        # Learn estimators from the target domain
        if self.verbose:
            print("Fit target domains")
        if train_target:
            self.target_estimator = self._fit_one_domain(
                self.target_train, task
            )

        else:
            self.target_estimator = self._fit_target_domain(self.target_train)
        self.fitted = True

    def _fit_target_domain(self, domain_data):
        # First fit m0 and cme_w_xz from the target domain

        # Fit the conditional mean embedding for each domain
        estimator = {}

        covars = {}
        covars["X"] = jnp.array(domain_data["X"])

        cme_w_x = ConditionalMeanEmbed(
            jnp.array(domain_data["W"]),
            covars,
            lam=self.lam_set["cme"],
            kernel_dict=self.kernel_dict["cme_w_x"],
            scale=self.sc,
            method=self.method_set["cme"],
            lam_min=self.lam_set["lam_min"],
            lam_max=self.lam_set["lam_max"],
            verbose=self.verbose,
        )

        estimator["cme_w_x"] = cme_w_x
        return estimator

    def evaluation(self, task="r", source_data=None, target_data=None):
        eval_list = []
        n_env = len(self.source_test)

        target_testx = {}
        if target_data is not None:
            target_testx["X"] = target_data[0]["X"]
            target_testy = target_data[0]["Y"]
        else:
            target_testx["X"] = self.target_test[0]["X"]
            target_testy = self.target_test[0]["Y"]

        for i in range(n_env):
            source_testx = {}

            if source_data is not None:
                source_testx["X"] = source_data[i]["X"]
                source_testy = source_data[i]["Y"]
            else:
                source_testx["X"] = self.source_test[i]["X"]
                source_testy = self.source_test[i]["Y"]

            # Source on source error
            predict_y = self.predict(source_testx, "source", "source", i)
            ss_error = self.score(predict_y, source_testy, task)
            eval_list.append(
                flatten(
                    {
                        "task": "source-source",
                        "env_id": int(i),
                        "predict error": ss_error,
                    }
                )
            )

            # Source on target error
            predict_y = self.predict(target_testx, "source", "source", i)
            st_error = self.score(predict_y, target_testy, task)
            eval_list.append(
                flatten(
                    {
                        "task": "source-target",
                        "env_id": int(i),
                        "predict error": st_error,
                    }
                )
            )

        # Target on target error
        predict_y = self.predict(target_testx, "target", "target", 0)
        tt_error = self.score(predict_y, target_testy, task)
        eval_list.append(
            flatten(
                {
                    "task": "target-target",
                    "env_id": 0 + n_env,
                    "predict error": tt_error,
                }
            )
        )

        # Adaptation error
        predict_y = self.predict(target_testx, "source", "target", 0)
        adapt_error = self.score(predict_y, target_testy, task)
        eval_list.append(
            flatten(
                {
                    "task": "adaptation",
                    "env_id": 0 + n_env,
                    "predict error": adapt_error,
                }
            )
        )

        df = pd.DataFrame(eval_list)
        if self.verbose:
            print(df)

        return df

    def predict(self, testX, k_domain, cme_domain, env_idx=0):
        if k_domain == "source":
            m0 = self.source_estimator["m0"]
        else:
            m0 = self.target_estimator["m0"]

        if cme_domain == "source":
            cme_w_x = self.source_estimator["cme_w_x"][env_idx]
        else:
            cme_w_x = self.target_estimator["cme_w_x"][0]

        predict_y = m0.get_exp_y_x(testX, cme_w_x)
        return predict_y

    def split_data(self):
        """Split training data."""
        n_env = len(self.source_train)
        train_list = [None, None, []]

        # Concatenate data from multiple environments
        for i in range(n_env):
            n = self.source_train[i]["X"].shape[0]
            index = np.random.RandomState(seed=42).permutation(n)
            split_id = np.split(index, [int(n / 6), int(n / 3)])

            if i == 0:
                for j, idx in enumerate(split_id):
                    if j < 2:
                        train_list[j] = split_data_widx(
                            self.source_train[i], idx
                        )

                    else:
                        train_list[2].append(
                            split_data_widx(self.source_train[i], idx)
                        )

            else:
                for j, idx in enumerate(split_id):
                    if j < 2:
                        train_list[j] = concatenate_data(
                            split_data_widx(self.source_train[i], idx),
                            train_list[j],
                        )

                    else:
                        train_list[2].append(
                            split_data_widx(self.source_train[i], idx)
                        )

        self.source_train = train_list

        n2 = self.target_train[0]["X"].shape[0]
        index = np.random.RandomState(seed=42).permutation(n2)
        split_id = np.split(index, [int(n2 / 3), int(n2 * 2 / 3)])
        train_list = []
        for j, idx in enumerate(split_id):
            if j == 2:
                train_list.append([split_data_widx(self.target_train[0], idx)])
            else:
                train_list.append(split_data_widx(self.target_train[0], idx))

        if self.verbose:
            print("Split data in target domain")
            print(
                "Number of training samples for cme(w|z,x):",
                train_list[0]["X"].shape[0],
            )
            print(
                "Number of training samples for h_0:",
                train_list[1]["X"].shape[0],
            )
            print(
                "Number of training samples for cme(w|x):",
                [d["X"].shape[0] for d in train_list[2]],
            )

        self.target_train = train_list
