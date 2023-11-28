"""
Adaptation pipeline of the deep kernel implementation.
"""

#Author: Katherine Tsai <kt14@illinois.edu>
#License: MIT


import torch
import logging
import json
from KPLA.models.deep_kernel.deep_model import DeepFullGraph
from KPLA.models.deep_kernel.trainer import FullDeepKernelTrainer
from KPLA.models.deep_kernel.torch_reg_utils import linear_reg_pred

logger = logging.getLogger()


class DeepKernelMethod:
  """ base estimator for the adaptation
  """
  def __init__(self, config_files, gpu_flg):
    """
    Args:
        data_config: .json file
        
    """
    with open(config_files, encoding="utf-8") as f:
      config = json.load(f)
      data_configs = config["data"]
      train_params = config["model"]
    self.fitted = False
    self.data_configs = data_configs
    self.train_params = train_params
    self.gpu_flg = gpu_flg


  def fit(self,
          source_traindata,
          target_traindata=None,
          split=False,
          verbose=2,
          plot=True):
    """fit the graph with deep kernel features.
    Args:
        source_traindata: dfaDataSetTorch
        target_traindata: dfaDataSetTorch or None
        split: split training data or not, Boolean
        verbose: int
        plot: plot loss or not, Boolean
    """

    model = self.trianer.train(source_traindata,
                               target_traindata,
                               split,
                               verbose,
                               plot)
    self.model = model

    self.w_dim = self.model.w_dim
    self.fitted = True

  def score(self, testy, predicty):
    """l2 error of the prediciton.
    Args:
        testy: true Y, torch.Tensor
        predicty: predict Y, torch.Tensor
    """
    ## Fix shape
    err_message = "unresolveable shape mismatch between test_y and predict_y"

    if testy.shape > predicty.shape:
      if not testy.ndim == predicty.ndim + 1:
        if not testy.shape[:-1] == predicty.shape:
          raise AssertionError(err_message)
      predicty = predicty.reshape(testy.shape)
    elif testy.shape < predicty.shape:
      if not testy.ndim + 1 == predicty.ndim:
        if not testy.shape == predicty.shape[:-1]:
          raise AssertionError(err_message)

      testy = testy.reshape(predicty.shape)
    l2_error =  torch.sum((testy-predicty)**2)/predicty.shape[0]
    return l2_error


  def predict(self):
    """Fits the model to the training data."""
    raise NotImplementedError("Implemented in child class.")

  def evaluation(self):
    """Fits the model to the training data."""
    raise NotImplementedError("Implemented in child class.")



class DeepFullAdapt(DeepKernelMethod):
  """implemnt full adaptation for deep kernel
  """
  def __init__(self, config_files, gpu_flg):

    DeepKernelMethod.__init__(self, config_files, gpu_flg)

    self.trianer = FullDeepKernelTrainer(self.data_configs,
                                         self.train_params,
                                         gpu_flg)
    #self.target_trianer = FullDeepKernelTrainer(self.data_configs,
    #                                             self.train_params,
    #                                             gpu_flg)

  def predict(self, testX, cme_wc_x_domain):
    """ predict the ourcome.
    Args:
        testX: covarites, torch.Tensor
        cme_wc_x_domain: specify domain, "original" or "adapt"
    """

    assert self.fitted is True
    if cme_wc_x_domain == "original":
      test_x1_feature = self.model.x1_source_feature_net(testX)
      tmp = DeepFullGraph.augment_single_feature(test_x1_feature,
                                                 self.model.add_cme_intercept)
      test_x1_feature = tmp
      ptest_wc1_feature = linear_reg_pred(test_x1_feature,
                                          self.model.source_coef_wc_x)

    elif cme_wc_x_domain == "adapt":
      test_x1_feature = self.model.x1_target_feature_net(testX)
      tmp = DeepFullGraph.augment_single_feature(test_x1_feature,
                                                 self.model.add_cme_intercept)
      test_x1_feature = tmp

      ptest_wc1_feature = linear_reg_pred(test_x1_feature,
                                          self.model.target_coef_wc_x)

    """
    ptest_w1_feature  = ptest_wc1_feature[:, 0:self.w_dim]
    ptest_c1_feature  = ptest_wc1_feature[:, self.w_dim::]
    
    feature = DeepFullGraph.augment_wc_feature(ptest_w1_feature,
                                                ptest_c1_feature,
                                                self.model.add_h0_intercept)
    pred = linear_reg_pred(feature, self.model.coef_h0)
    """
    pred = linear_reg_pred(ptest_wc1_feature, self.model.coef_h0)

    return pred
