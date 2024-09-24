

**KPLA** is a python implementation of the paper, ``Proxy methods for domain adaptation''. 


## Installation
### Dependencies
KPLA requires:

- cosde [download link](https://github.com/tsai-kailin/ConditionalOSDE)
- latent_shift_adaptation [download link](https://github.com/google-research/google-research/tree/master/latent_shift_adaptation/latent_shift_adaptation)
- cvxpy (>= 1.4.1)
- cvxopt (>= 1.3.2)
- jax (>= 0.4.25)
- pandas (>= 2.0.3)
- matplotlib (>= 3.7.2)
- numpy (>= 1.24.3)
- scikit-image (>= 0.22.0)
- scikit-learn (>= 1.3.0)
- scipy (>= 1.11.2)
- tqdm (>= 4.66.1)
- tensorflow (>= 2.11.0)      


=======


Install the KPLA
```
python setup.py install
```


## Usage
#### Steps to run the experiment of adaptation with concepts and proxies:
1. Prepare data in the following dictionary form:
```
  data = {
    "X": jax.numpy.ndarray, (n_samples, n_features) or (n_samples,)
    "Y": jax.numpy.ndarray, (n_samples, n_features) or (n_samples,)
    "W": jax.numpy.ndarray, (n_samples, n_features) or (n_samples,)
    "C": jax.numpy.ndarray, (n_samples, n_features) or (n_samples,)
  }
```
2. Specify the index of the kernel function (string) of each variable.  
```
  kernel_dict = {}

  kernel_dict["cme_w_xc"] = {"X":  KERNEL_X,
                            "C":   KERNEL_C,
                            "Y":   KERNEL_W} #Y is W
  kernel_dict["cme_wc_x"] = {"X":  KERNEL_X,
                            "Y": [{"kernel": KERNEL_W, "dim": DIM_W},            # 
                                  {"kernel": KERNEL_C, "dim": DIM_C}]} # Y is (W,C)
  kernel_dict["h0"]       = {"C": KERNEL_C}
```
Current implementation of the kernel function:
| Kernel function           | Index        | 
|---------------------------|--------------|
|radial basis function (RBF)|"rbf"         | 
|columnwise RBF             |"rbf_column"  |
|binary                     |"binary"      |
|columnwise binary          |"binary_column|

3. Prepare method set and lambda (regualrization) set:
```
  method_set = {"cme": "original", "h0": "original"}
  lam_set = {"cme":    LAM_1,   # L2 penalty for the conditional mean embedding
            "h0":      LAM_2,   # L2 penalty for the bridge function
            "lam_min": LAM_MIN, 
            "lam_max": LAM_MAX}
```
Note: the current version only implements method "original" which computes the whole Gram matrix. We plan to implement Nystrom or other approximation method in the future.
4.  Train the model
```
from KPLA.models.plain_kernel.adaptation import FullAdapt

estimator_full = FullAdapt(source_train,
                           target_train,
                           source_test,
                           target_test,
                           split,       # split the training data or not, Boolean
                           scale,       # kernel length-scale, float
                           lam_set,
                           method_set,
                           kernel_dict)

estimator_full.fit(task = TASK) # task="c" for classification, task="r" for regression
estimator_full.evaluation(task = TASK)
```
5. Model selection using cross-validation or validation set.
We implement the function to select the parameters of `lam_set` and `scale`.
```
from KPLA.models.plain_kernel.model_selection import tune_adapt_model_cv

b_estimator, b_params = tune_adapt_model_cv(source_train,
                                            target_train,
                                            source_test,
                                            target_test,
                                            method_set,
                                            kernel_dict,
                                            use_validation = USE_VAL, # True: extra validation set False:cross-validation 
                                            val_data,       
                                            model          = FullAdapt,          
                                            task           = TASK,
                                            fit_task       = TASK,
                                            n_params       = N_PARAMS, 
                                            n_fold         = N_FOLD,
                                            min_log        = MIN_VAL, # minimum value of grid search, log10 scale
                                            max_log        = MAX_VAL, # maximum value of grid search, log10 scale
                                            )
```

#### Steps to run the experiment of multi-source adaptation:
1.  Prepare data list in the following dictionary form:
```
  data_domain_i = {
    "X": jax.numpy.ndarray, (n_samples, n_features) or (n_samples,)
    "Y": jax.numpy.ndarray, (n_samples, n_features) or (n_samples,)
    "W": jax.numpy.ndarray, (n_samples, n_features) or (n_samples,)
    "Z": jax.numpy.ndarray, (n_samples, n_domains) or (n_samples,) # domain index, every entry has the same value
  }
  data = [data_domain_1, ..., data_domain_n]
```
2. Specify the index of the kernel function (string) of each variable. There are two versions of the multi-source adaptation. 


To use `MultiEnvAdaptCAT`, the kernel script is:
```
kernel_dict = {}

kernel_dict['cme_w_xz'] = {'X': KERNEL_X, 'Y': KERNEL_W} # Y is W
kernel_dict['cme_w_x']  = {'X': KERNEL_X, 'Y': KERNEL_W} # Y is W
kernel_dict['m0']       = {'X': KERNEL_X}

```

To use `MultiEnvAdapt`, the kernel script is:
```
kernel_dict = {}

kernel_dict['cme_w_xz'] = {'X': KERNEL_X, 'Y': KERNEL_W, 'Z': KERNEL_Z} # Y is W
kernel_dict['cme_w_x']  = {'X': KERNEL_X, 'Y': KERNEL_W}                # Y is W
kernel_dict['m0']       = {'X': KERNEL_X}
```
Let $Z$ be the domain index and is one-hot encoded. Setting `KENEL_Z='binary'` in `MultiEnvAdapt` is the same as using `MultiEnvAdaptCAT`. `MultiEnvAdapt` is a more flexible version of `MultiEnvAdaptCAT` that can take continuous value of $Z$ and allows user to specify the underlying kernel function. 
3. Prepare method set and lambda (regualrization) set:
```
  method_set = {"cme": "original", "m0": "original"}
  lam_set = {"cme":     LAM_1,   # L2 penalty for the conditional mean embedding
             "m0":      LAM_2,   # L2 penalty for the bridge function
             "lam_min": LAM_MIN, 
             "lam_max": LAM_MAX}
```
4. Train the model
```
from KPLA.models.plain_kernel.multienv_adaptation import  MultiEnvAdaptCAT


```

```
from KPLA.models.plain_kernel.multienv_adaptation import  MultiEnvAdapt
estimator_multi_a = MultiEnvAdapt(source_train,
                                  target_train,
                                  source_test,
                                  target_test,
                                  split,
                                  scale,
                                  lam_set,
                                  method_set,
                                  kernel_dict)
```
5. Model selection using cross-validation or validation set.



## Run Experiments
Navigate the examples in `./tests` directory. 

First execute the model selection program under `./model_selection` then run the program under `./experiments` folder.

For the simulated regression tasks, `sim_multisource_bin` and `sim_multisource_cont`, first launch execute the hyperparameter tuning program `test_proposed_onehot.py` for each regression task. Then run the experiments for the baseline (`sweep_baselines.py`) and proposed (`sweep_proposed.py`) methods.


## References

Please use the following metadata for the citation.

```bibtex
@inproceedings{tsai2024proxy,
  title={Proxy Methods for Domain Adaptation},
  author={Tsai, Katherine and Pfohl, Stephen R and Salaudeen, Olawale and Chiou, Nicole and Kusner, Matt and D’Amour, Alexander and Koyejo, Sanmi and Gretton, Arthur},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={3961--3969},
  year={2024},
  organization={PMLR}
}
```



