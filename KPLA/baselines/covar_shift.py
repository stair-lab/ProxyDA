"""
implementation of covariate shift adaptation method.
"""
#Author: Nicole Chiou, Katherine Tsai<kt14@illinois.edu>
#MIT LICENSE


import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.kernel_ridge import KernelRidge
from sklearn import random_projection as r_proj



def convert_data_Y2D(source, target):
  source_Y = np.zeros_like(source['Y'])
  target_Y = np.ones_like(target['Y'])

  return {
      'X': np.concatenate([source['X'], target['X']], axis=0),
      'Y': np.concatenate([source_Y, target_Y], axis=0).ravel(),
  }

class COVAR:
  """
  implementation of covariate shifts
  """
  def __init__(self, alpha=1.0, kernel='rbf', proj_dim=16):
    self.source_covar_model = KernelRidge(alpha=alpha, kernel=kernel)
    self.target_covar_model = KernelRidge(alpha=alpha, kernel=kernel)
    self.proj_dim = proj_dim

  def fit(self, source_train, target_train):


    domain_d = convert_data_Y2D(source_train, target_train)
    proj = r_proj.GaussianRandomProjection(n_components=self.proj_dim,
                                          random_state=0).fit(domain_d['X'])
    d_x = LogisticRegression(random_state=0)
    d_x.fit(proj.transform(domain_d['X']), domain_d['Y'])

    # Compute sample weights
    q_x_train = d_x.predict_proba(proj.transform(source_train['X']))[:, 1]
    source_sample_weight_train = q_x_train / (1. - q_x_train + 1e-3)

    #fit source model
    self.source_covar_model.fit(source_train['X'],
                                source_train['Y'],
                                sample_weight=source_sample_weight_train)

    # compute sample weights
    q_x_train = d_x.predict_proba(proj.transform(target_train['X']))[:, 0]
    target_sample_weight_train = q_x_train / (1. - q_x_train + 1e-3)
    #fit target model
    self.target_covar_model.fit(target_train['X'],
                                target_train['Y'],
                                sample_weight=target_sample_weight_train)



  def predict(self, test_data):
    return self.source_covar_model.predict(test_data['X'])

  def predict_target(self, test_data):
    return self.target_covar_model.predict(test_data['X'])
