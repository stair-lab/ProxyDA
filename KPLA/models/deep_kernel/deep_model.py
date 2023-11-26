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

class DeepFullGraph:
  """
  Adaptation setting: observe (W,X,Y,C) from the source, (W,X,C) from the target

  """
  def __init__(self,
                x1_source_feature_net: nn.Module,
                x1_target_feature_net: nn.Module,
                x2_feature_net: nn.Module,
                #w1_feature_net: nn.Module,
                w2_feature_net: nn.Module,
                #c1_feature_net: nn.Module,
                c2_feature_net: nn.Module,
                c3_feature_net: nn.Module,
                add_cme_intercept: bool,
                add_h0_intercept: bool
                ):

    self.x1_source_feature_net = x1_source_feature_net

    if x1_target_feature_net is not None:
      self.x1_target_feature_net = x1_target_feature_net

    self.x2_feature_net = x2_feature_net #w_xc
    #self.w1_feature_net = w1_feature_net
    self.w2_feature_net = w2_feature_net #w_xc
    #self.c1_feature_net = c1_feature_net
    self.c2_feature_net = c2_feature_net #w_xc
    self.c3_feature_net = c3_feature_net #h0

    self.add_cme_intercept = add_cme_intercept
    self.add_h0_intercept = add_h0_intercept
  @staticmethod
  def augment_single_feature(feature, add_intercept):
    if add_intercept:
      feature = add_const_col(feature)

    return feature

  @staticmethod
  def augment_xc_feature(x_feature, c_feature, add_intercept):
    if add_intercept:
      x_feature = add_const_col(x_feature)
      c_feature = add_const_col(c_feature)

    feature = outer_prod(x_feature, c_feature)
    feature = torch.flatten(feature, start_dim=1)
    return feature

  @staticmethod
  def augment_wc_feature(w_feature: torch.Tensor,
                          c_feature: torch.Tensor,
                          add_intercept: bool):

    if add_intercept:
      w_feature = add_const_col(w_feature)
      c_feature = add_const_col(c_feature)


    feature = outer_prod(c_feature, w_feature )
    feature = torch.flatten(feature, start_dim=1)

    return feature

  @staticmethod
  def fit_h0(x2_feature1: torch.Tensor,
                x2_feature2: torch.Tensor,
                c2_feature1: torch.Tensor,
                c2_feature2: torch.Tensor,
                c3_feature2: torch.Tensor,
                w2_feature1: torch.Tensor,
                #w2_feature2: Optional[torch.Tensor],
                y2:          torch.Tensor,
                cme_lam:            float,
                h0_lam:             float,
                add_cme_intercept:   bool,
                add_h0_intercept:    bool,
                ):

    # stage1
    feature =  DeepFullGraph.augment_xc_feature(x2_feature1,
                                                c2_feature1,
                                                add_cme_intercept)
    beta = fit_linear(w2_feature1, feature, cme_lam)

    # predicting for stage 2
    feature = DeepFullGraph.augment_xc_feature(x2_feature2,
                                               c2_feature2,
                                               add_cme_intercept)

    predicted_w_feature2 = linear_reg_pred(feature, beta)

    # stage2
    feature = DeepFullGraph.augment_wc_feature(predicted_w_feature2,
                                                c3_feature2,
                                                add_h0_intercept)

    alpha = fit_linear(y2, feature, h0_lam)


    pred = linear_reg_pred(feature, alpha)
    tmp1 = torch.norm((y2 - pred)) ** 2/y2.shape[0]
    stage2_loss =  tmp1 + h0_lam * torch.norm(alpha) ** 2

    #mean_w2_feature2 = None
    #if w2_feature2 is not None:
    #    mean_w2_feature2=torch.mean(mean_w2_feature2, dim=0, keepdim=True)

    return dict(beta=beta,
                predicted_w_feature2=predicted_w_feature2,
    #            mean_w2_feature = mean_w2_feature2,
                alpha=alpha,
                loss=stage2_loss)
  """
  @staticmethod
  def fit_cme_wc_x(x1_feature1: torch.Tensor,
                  x1_feature2: torch.Tensor,
                  wc1_feature1: torch.Tensor,
                  y2:           torch.Tensor,
                  alpha:        torch.Tensor,
                  w_dim:                 int,
                  cme_lam:             float, 
                  h0_lam:              float,
                  add_cme_intercept:    bool,
                  add_h0_intercept:     bool,
                  ):

      # stage1
      beta = fit_linear(wc1_feature1, x1_feature1, cme_lam)

      # predicting for stage 2
      predicted_wc1_feature2 = linear_reg_pred(x1_feature2, beta)

      predicted_w1_feature2  = predicted_wc1_feature2[:, 0:w_dim]
      predicted_c1_feature2  = predicted_wc1_feature2[:, w_dim::]

      # stage2
      feature = DeepFullGraph.augment_wc_feature(predicted_w1_feature2,
                                                    predicted_c1_feature2,
                                                  add_h0_intercept)
      #print(feature.shape, alpha.shape)
      
      
      
      #alpha = fit_linear(y2, feature, h0_lam) #alpha shall be from h0
      pred = linear_reg_pred(feature, alpha)
      stage2_loss = torch.norm((y2 - pred)) ** 2/y2.shape[0] #+ cme_lam * torch.norm(beta) ** 2


      return dict(beta=beta,
                  loss=stage2_loss)
  """
  def fit_t(self,
        train_data1: dfaDataSetTorch,
        train_data2: dfaDataSetTorch,
        train_data3: dfaDataSetTorch,
        target_data: Optional[dfaDataSetTorch],
        cme_lam:     float,
        h0_lam:      float):

    x2_feature2 = self.x2_feature_net(train_data2.X) #cme W_xc
    x2_feature3 = self.x2_feature_net(train_data3.X) #cme W_xc
    c2_feature2 = self.c2_feature_net(train_data2.C) #cme W_xc
    c2_feature3 = self.c2_feature_net(train_data3.C) #cme W_xc
    w2_feature2 = self.w2_feature_net(train_data2.W) #cme W_xc


    c3_feature3 = self.c3_feature_net(train_data3.C) #h0

    w2_feature1 = self.w2_feature_net(train_data1.W) #cme WC_x
    c3_feature1 = self.c3_feature_net(train_data1.C) #cme WC_x
    x1_feature1 = self.x1_source_feature_net(train_data1.X) #cme WC_x
    tmp = DeepFullGraph.augment_single_feature(x1_feature1,
                                               self.add_cme_intercept) #cme WC_x
    x1_feature1 = tmp
    """
    if len(w2_feature1.shape) == 1:
        w2_feature1 = w2_feature1[:, None]
    if len(c3_feature1.shape) == 1:
        c3_feature1 = c3_feature1[:, None]
    
    w2c3_feature1 = torch.cat((w2_feature1, c3_feature1), 1)
    """

    w2c3_feature1 = DeepFullGraph.augment_wc_feature(w2_feature1,
                                                     c3_feature1,
                                                     self.add_h0_intercept)

    if target_data is not None and self.x1_target_feature_net:

      x1_target_feature = self.x1_target_feature_net(target_data.X) #cme WC_x
      tmp = DeepFullGraph.augment_single_feature(x1_target_feature,
                                                 self.add_cme_intercept)
      x1_target_feature = tmp

      w2_target_feature = self.w2_feature_net(target_data.W) #cme WC_x
      c3_target_feature = self.c3_feature_net(target_data.C) #cme WC_x
      """
      if len(w2_target_feature.shape) == 1:
          w2_target_feature = w2_target_feature[:, None]
      if len(c3_target_feature.shape) == 1:
          c3_target_feature = c3_target_feature[:, None]
      
      w2c3_target_feature = torch.cat((w2_target_feature, c3_target_feature), 1)
      """
      w2c3_target_feature = DeepFullGraph.augment_wc_feature(w2_target_feature,
                                                        c3_target_feature,
                                                        self.add_h0_intercept)

    w_dim = w2_feature2.shape[1]
    self.w_dim = w_dim

    res = self.fit_h0(x2_feature2,
              x2_feature3,
              c2_feature2,
              c2_feature3,
              c3_feature3,
              w2_feature2,
              train_data3.Y,
              cme_lam,
              h0_lam,
              self.add_cme_intercept,
              self.add_h0_intercept,
              )


    beta_source = fit_linear(w2c3_feature1,
                             x1_feature1,
                             cme_lam)

    if self.x1_target_feature_net and target_data is not None:
      beta_target = fit_linear(w2c3_target_feature,
                               x1_target_feature,
                               cme_lam)
      self.target_coef_wc_x = beta_target


    self.coef_h0   = res['alpha']
    self.source_coef_wc_x = beta_source



  def fit(self,
          train_data1,
          train_data2,
          train_data3,
          target_data,
          cme_lam,
          h0_lam):
    #convert data to torch.Tensor
    train_data1 = dfaDataSetTorch.from_jaxnumpy(train_data1)
    train_data2 = dfaDataSetTorch.from_jaxnumpy(train_data2)
    train_data3 = dfaDataSetTorch.from_jaxnumpy(train_data3)
    if target_data is not None:
      target_data = dfaDataSetTorch.from_jaxnumpy(target_data)

    self.fit_t(train_data1,
               train_data2,
               train_data3,
               target_data,
               cme_lam,
               h0_lam)

  def predict_t(self, testx, domain='source'):
    #find the mean embedding of cme_wc_xnew
    if domain == 'source':
      test_x1_feature =self.x1_source_feature_net(testx)
      tmp = DeepFullGraph.augment_single_feature(test_x1_feature,
                                                 self.add_cme_intercept)
      test_x1_feature = tmp
      ptest_wc1_feature = linear_reg_pred(test_x1_feature,
                                          self.source_coef_wc_x)
    else:
      test_x1_feature =self.x1_target_feature_net(testx)
      tmp = DeepFullGraph.augment_single_feature(test_x1_feature,
                                                 self.add_cme_intercept)
      test_x1_feature = tmp
      ptest_wc1_feature = linear_reg_pred(test_x1_feature,
                                          self.target_coef_wc_x)
    """
    ptest_w1_feature  = ptest_wc1_feature[:, 0:self.w_dim]
    ptest_c1_feature  = ptest_wc1_feature[:, self.w_dim::]

    
    feature = DeepFullGraph.augment_wc_feature(ptest_w1_feature,
                                                ptest_c1_feature,
                                                add_h0_intercept)
    """
    #pred = linear_reg_pred(feature, self.coef_h0)
    pred = linear_reg_pred(ptest_wc1_feature, self.coef_h0)
    return pred

  def predict(self, testx, domain='source'):
    testx = torch.tensor(np.asarray(testx), dtype=torch.float32)
    return self.predict_t(testx, domain).data.numpy()

  def evaluate_t(self, test_data, domain='source'):
    truey = test_data.Y
    with torch.no_grad():
      pred = self.predict_t(test_data.X, domain)
    return torch.mean((truey - pred) ** 2)

  def evaluate(self, test_data, domain='source'):
    return self.evaluate_t(dfaDataSetTorch.from_jaxnumpy(test_data),
                           domain).data.item()

