"""
Base estimator of the deep kernel implementation.
"""

#Author: Katherine Tsai <kt14@illinois.edu>
#License: MIT

import numpy as np
# import jax.numpy as jnp

from typing import Optional


import torch
from torch import nn
import logging
from KPLA.models.deep_kernel.torch_reg_utils import add_const_col, outer_prod, fit_linear, linear_reg_pred
from KPLA.data.data_class import dfaDataSetTorch

logger = logging.getLogger()

class DeepMultiEnvGraph:
  """
  Adaptation setting: observe (W,X,Y,E) from the source, (W,X) from the target

  """
  def __init__(self,
                x1_target_feature_net: nn.Module,
                x2_feature_net: nn.Module,
                x3_feature_net: nn.Module,
                w2_feature_net: nn.Module,
                e2_discrete: bool,
                e2_feature_net: Optional[nn.Module],
                add_cme_intercept: bool,
                add_m0_intercept: bool
                ):


    if x1_target_feature_net is not None:
      self.x1_target_feature_net = x1_target_feature_net

    self.x2_feature_net = x2_feature_net #w_xe
    self.x3_feature_net = x3_feature_net #m0
    self.w2_feature_net = w2_feature_net #w_xe
    self.e2_discrete = e2_discrete
    if not self.e2_discrete:
      self.e2_feature_net = e2_feature_net #w_xe
    else:
      self.e2_feature_net = None

    self.add_cme_intercept = add_cme_intercept
    self.add_m0_intercept = add_m0_intercept
  @staticmethod
  def augment_single_feature(feature, add_intercept):
    if add_intercept:
      feature = add_const_col(feature)

    return feature

  @staticmethod
  def augment_xe_feature(x_feature, e_feature, add_intercept):
    if add_intercept:
      x_feature = add_const_col(x_feature)
      e_feature = add_const_col(e_feature)

    feature = outer_prod(x_feature, e_feature)
    feature = torch.flatten(feature, start_dim=1)
    return feature

  @staticmethod
  def augment_wx_feature(w_feature, x_feature, add_intercept):
    if add_intercept:
      w_feature = add_const_col(w_feature)
      x_feature = add_const_col(x_feature)

    feature = outer_prod(w_feature, x_feature)
    feature = torch.flatten(feature, start_dim=1)
    return feature

  @staticmethod
  def fit_m0(x2_feature1: torch.Tensor,
            x2_feature2: torch.Tensor,
            x3_feature2: torch.Tensor,
            e2_feature1: torch.Tensor,
            e2_feature2: torch.Tensor,
            w2_feature1: torch.Tensor,
            y2:          torch.Tensor,
            cme_lam:            float,
            m0_lam:             float,
            add_cme_intercept:   bool,
            add_m0_intercept:    bool,
            e2_discrete:         bool,
            ):

    # stage1
    if not e2_discrete:
      feature =  DeepMultiEnvGraph.augment_xe_feature(x2_feature1,
                                                      e2_feature1,
                                                      add_cme_intercept)
      beta = fit_linear(w2_feature1, feature, cme_lam)

      # predicting for stage 2
      feature = DeepMultiEnvGraph.augment_xe_feature(x2_feature2,
                                                     e2_feature2,
                                                     add_cme_intercept)

      predicted_w_feature2 = linear_reg_pred(feature, beta)
    else:
      beta_dict = {}
      #sort w based on e
      unique_e, unique_id = torch.unique(e2_feature1,
                                         sorted=True,
                                         return_inverse=True)
      for e_val, i in enumerate(unique_e):
        con = unique_id==i
        select_id = con.nonzero()

        beta = fit_linear(w2_feature1[select_id,...],
                          x2_feature1[select_id,...],
                          cme_lam)
        beta_dict[e_val]= beta

      #predicting for stage 2
      predicted_w_feature2 = torch.zeros((e2_feature2.size(dim=0),
                                          w2_feature1.size(dim=1)))
      unique_e, unique_id = torch.unique(e2_feature2,
                                          sorted=True,
                                          return_inverse=True)
      for e_val, i in enumerate(unique_e):
        con = unique_id==i
        select_id = con.nonzero()
        pre_w = linear_reg_pred(x2_feature2[select_id,...],
                                beta_dict[e_val])
        predicted_w_feature2[select_id,...] = pre_w

    # stage2
    feature = DeepMultiEnvGraph.augment_wx_feature(predicted_w_feature2,
                                                    x3_feature2,
                                                    add_m0_intercept)

    alpha = fit_linear(y2, feature, m0_lam)


    pred = linear_reg_pred(feature, alpha)
    tmp1 = torch.norm((y2 - pred)) ** 2/y2.shape[0]
    stage2_loss =  tmp1 + m0_lam * torch.norm(alpha) ** 2

    return dict(beta=beta,
                predicted_w_feature2=predicted_w_feature2,
    #            mean_w2_feature = mean_w2_feature2,
                alpha=alpha,
                loss=stage2_loss)

  def fit_t(self,
        train_data2: dfaDataSetTorch,
        train_data3: dfaDataSetTorch,
        target_data: Optional[dfaDataSetTorch],
        cme_lam:     float,
        m0_lam:      float):

    x2_feature2 = self.x2_feature_net(train_data2.X) #cme W_xe
    x2_feature3 = self.x2_feature_net(train_data3.X) #cme W_xe
    if self.e2_discrete:
      e2_feature2 = train_data2.E
      e2_feature3 = train_data3.E
    else:
      e2_feature2 = self.e2_feature_net(train_data2.E) #cme W_xe
      e2_feature3 = self.e2_feature_net(train_data3.E) #cme W_xe

    w2_feature2 = self.w2_feature_net(train_data2.W) #cme W_xe


    x3_feature3 = self.x3_feature_net(train_data3.X) #m0

    if target_data is not None and self.x1_target_feature_net:

      x1_target_feature = self.x1_target_feature_net(target_data.X) #cme W_x
      tmp = DeepMultiEnvGraph.augment_single_feature(x1_target_feature,
                                                 self.add_cme_intercept)
      x1_target_feature = tmp

      w2_target_feature = self.w2_feature_net(target_data.W) #cme WC_x

    w_dim = w2_feature2.shape[1]
    self.w_dim = w_dim

    res = self.fit_m0(x2_feature2,
                x2_feature3,
                x3_feature3,
                e2_feature2,
                e2_feature3,
                w2_feature2,
                train_data3.Y,
                cme_lam,
                m0_lam,
                self.add_cme_intercept,
                self.add_m0_intercept,
                self.e2_discrete
              )

    if self.x1_target_feature_net and target_data is not None:
      beta_target = fit_linear(w2_target_feature,
                               x1_target_feature,
                               cme_lam)
      self.target_coef_w_x = beta_target


    self.coef_m0   = res['alpha']



  def fit(self,
          train_data2,
          train_data3,
          target_data,
          cme_lam,
          m0_lam):
    #convert data to torch.Tensor
    train_data2 = dfaDataSetTorch.from_jaxnumpy(train_data2)
    train_data3 = dfaDataSetTorch.from_jaxnumpy(train_data3)
    if target_data is not None:
      target_data = dfaDataSetTorch.from_jaxnumpy(target_data)

    self.fit_t(
               train_data2,
               train_data3,
               target_data,
               cme_lam,
               m0_lam)

  def predict_t(self, testx):
    #find the mean embedding of cme_wc_xnew

    test_x1_feature =self.x1_target_feature_net(testx)
    tmp = DeepMultiEnvGraph.augment_single_feature(test_x1_feature,
                                                   self.add_cme_intercept)
    test_x1_feature = tmp

    ptest_w1_feature = linear_reg_pred(test_x1_feature,
                                      self.target_coef_w_x)
    ptest_wx1_feature =  DeepMultiEnvGraph.augment_wx_feature(ptest_w1_feature,
                                                        test_x1_feature,
                                                        self.add_cme_intercept)
    pred = linear_reg_pred(ptest_wx1_feature, self.coef_m0)
    return pred

  def predict(self, testx):
    testx = torch.tensor(np.asarray(testx), dtype=torch.float32)
    return self.predict_t(testx).data.numpy()

  def evaluate_t(self, test_data):
    truey = test_data.Y
    with torch.no_grad():
      pred = self.predict_t(test_data.X)
    return torch.mean((truey - pred) ** 2)

  def evaluate(self, test_data):
    return self.evaluate_t(dfaDataSetTorch.from_jaxnumpy(test_data)).data.item()

