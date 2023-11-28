"""
Adaptation pipeline of the multi-source deep kernel implementation.
"""

#Author: Katherine Tsai <kt14@illinois.edu>
#License: MIT


import logging
from  KPLA.models.deep_kernel.deep_adaptation import DeepKernelMethod
from  KPLA.models.deep_kernel.multienv_deep_model import DeepMultiEnvGraph
from  KPLA.models.deep_kernel.multienv_trainer import MultiEnvDeepKernelTrainer
from  KPLA.models.deep_kernel.torch_reg_utils import linear_reg_pred

logger = logging.getLogger()


class DeepMultiEnvAdapt(DeepKernelMethod):
  """implemnt multi-source adaptation for deep kernel
  """
  def __init__(self, config_files, gpu_flg):

    DeepKernelMethod.__init__(self, config_files, gpu_flg)

    self.trianer = MultiEnvDeepKernelTrainer(self.data_configs,
                                              self.train_params,
                                              gpu_flg)

  def predict(self, testX):
    """ predict the ourcome.
    Args:
        testX: covarites, torch.Tensor
    """

    assert self.fitted is True

    test_x1_feature = self.model.x1_target_feature_net(testX)
    tmp = DeepMultiEnvGraph.augment_single_feature(test_x1_feature,
                                                   self.model.add_cme_intercept)
    test_x1_feature = tmp
    ptest_w1_feature = linear_reg_pred(test_x1_feature,
                                        self.model.target_coef_w_x)
    test_x3_feature = self.model.x3_feature_net(testX)
    feature = DeepMultiEnvGraph.augment_wx_feature(ptest_w1_feature,
                                                  test_x3_feature,
                                                  self.model.add_m0_intercept)
    pred = linear_reg_pred(feature, self.model.coef_m0)

    return pred
