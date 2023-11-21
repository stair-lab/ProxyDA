
import numpy as np
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import make_scorer, mean_squared_error, roc_auc_score
from sklearn.gaussian_process.kernels import RBF

from itertools import product
import time
from multiprocessing import Pool, cpu_count


def select_kernel_ridge_model(model, X, y, sample_weight=None, n_params=8, n_fold=5, min_val=-4, max_val=3):
  param_grid = {
      'gamma': np.logspace(min_val, max_val, n_params),  # Adjust the range as needed
      'alpha': np.logspace(min_val, max_val, n_params),   # Adjust the range as needed
  }

  scorer = make_scorer(mean_squared_error, greater_is_better=False)
  kf = KFold(n_splits=n_fold, shuffle=True, random_state=42)
  grid_search = GridSearchCV(model, param_grid, cv=kf, scoring=scorer, n_jobs=150)
  grid_search.fit(X, y, sample_weight=sample_weight)

  print("Best Parameters: ", grid_search.best_params_)

  return grid_search.best_estimator_, grid_search.best_params_



