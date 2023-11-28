"""
Trainer of the multienv deep kernel method.
"""

#Author: Katherine Tsai <kt14@illinois.edu>
#License: MIT

from typing import Dict, Any, Optional
import torch
from torch import nn
import logging


from KPLA.models.deep_kernel.torch_reg_utils import linear_reg_loss
from KPLA.models.deep_kernel.multienv_deep_model import DeepMultiEnvGraph
from KPLA.models.deep_kernel.nn_structure import build_extractor
from KPLA.data.data_class import  multi_split_train_data
import matplotlib.pyplot as plt
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class MultiEnvDeepKernelTrainer:
  """trainer for deep kernel adaptation method.
  """
  def __init__(self,
               data_configs: Dict[str, Any],
               train_params: Dict[str, Any],
               gpu_flg: bool = False):

    self.track_loss = {"m0":[],
                       "w_xe":[],
                       "w_x.x_target":[],
                       }

    self.data_config = data_configs
    self.gpu_flg = gpu_flg and torch.cuda.is_available()
    if self.gpu_flg:
      logger.info("gpu mode")
    # configure training params

    self.lam_set: dict = train_params["lam_set"]
    self.cme_iter: int = train_params["cme_iter"]
    self.m0_iter: int = train_params["m0_iter"]

    self.n_epoch: int = train_params["n_epoch"]
    self.add_cme_intercept = True
    self.add_m0_intercept = True
    self.w_weight_decay = train_params["w_weight_decay"]
    self.x_weight_decay = train_params["x_weight_decay"]
    self.e_weight_decay = train_params["e_weight_decay"]
    self.e2_discrete    = train_params["e2_discrete"]

    self.w_lr = train_params["w_lr"]
    self.x_lr = train_params["x_lr"]
    self.e_lr = train_params["e_lr"]


    # build networks
    networks = build_extractor(data_configs["name"])
    self.x1_target_feature_net:  Optional[nn.Module] = networks[0]

    self.x2_feature_net: nn.Module = networks[1]
    self.x3_feature_net: nn.Module = networks[2]

    self.w2_feature_net: nn.Module = networks[3]

    if not self.e2_discrete:
      self.e2_feature_net: nn.Module = networks[4]


    if self.gpu_flg:

      if self.x1_target_feature_net is not None:
        self.x1_target_feature_net.to("cuda:0")
      self.x2_feature_net.to("cuda")
      self.x3_feature_net.to("cuda")

      self.w2_feature_net.to("cuda")
      if not self.e2_discrete:
        self.e2_feature_net.to("cuda")

    if self.x1_target_feature_net:
      target_opt = torch.optim.Adam(self.x1_target_feature_net.parameters(),
                                    weight_decay=self.x_weight_decay,
                                    lr=self.x_lr)
      self.x1_target_opt = target_opt

    self.x2_opt = torch.optim.Adam(self.x2_feature_net.parameters(),
                                    weight_decay=self.x_weight_decay,
                                    lr=self.x_lr)

    self.x3_opt = torch.optim.Adam(self.x3_feature_net.parameters(),
                                    weight_decay=self.x_weight_decay,
                                    lr=self.x_lr)

    self.w2_opt = torch.optim.Adam(self.w2_feature_net.parameters(),
                                    weight_decay=self.w_weight_decay,
                                    lr=self.w_lr)

    if not self.e2_discrete:

      self.e2_opt = torch.optim.Adam(self.e2_feature_net.parameters(),
                                      weight_decay=self.e_weight_decay,
                                      lr=self.e_lr)



  def train(self,
            source_traindata,
            target_traindata,
            split,
            verbose,
            plot=True):
    """
    Args:
        source_traindata: dfaDataSetTorch
        target_traindata: dfaDataSetTorch
        split: Boolean
        verbose: int
        plot: Boolean
    """
    if split:
      train_set = multi_split_train_data(source_traindata, 2)
      source_traindata2 = train_set[0]
      source_traindata3 = train_set[1]
      if self.gpu_flg:
        source_traindata2 = source_traindata2.to_gpu()
        source_traindata3 = source_traindata3.to_gpu()
        if target_traindata is not None:
          target_traindata = target_traindata.to_gpu()

      for t in range(self.n_epoch):
        self.cme_w_xe_feature_update(source_traindata2,
                                     verbose)
        self.m0_feature_update(source_traindata2,
                               source_traindata3,
                               verbose)
        self.cme_w_x_feature_update(target_traindata,
                                     verbose)

        if verbose >= 1:
          logger.info(f"Epoch {t} ended")


    else:
      if self.gpu_flg:
        source_traindata = source_traindata.to_gpu()

      for t in range(self.n_epoch):
        self.cme_w_xe_feature_update(source_traindata,
                                      verbose)
        self.m0_feature_update(source_traindata,
                                source_traindata,
                                verbose)
        self.cme_w_x_feature_update( target_traindata,
                                      verbose)
        if verbose >= 1:
          logger.info(f"Epoch {t} ended")

    if plot:
      plt.figure()
      plt.plot(self.track_loss["m0"], label="m0")
      plt.xlabel("iteration")
      plt.ylabel("m0 loss")
      plt.yscale("log")
      plt.savefig("m0_loss.png", bbox_inches="tight")

      plt.figure()
      plt.plot(self.track_loss["w_xe"], label="w_xe")
      plt.xlabel("iteration")
      plt.ylabel("w_xe loss")
      plt.yscale("log")
      plt.savefig("cme_w_xe_loss.png", bbox_inches="tight")


      plt.figure()
      plt.plot(self.track_loss["w_x.x_target"], label="w_x.x")
      plt.xlabel("iteration")
      plt.ylabel("w_x.x_target loss")
      plt.yscale("log")
      plt.savefig("cme_w_x_x_target.png", bbox_inches="tight")


    #need to be completed
    dfa = DeepMultiEnvGraph(self.x1_target_feature_net,
                            self.x2_feature_net,
                            self.x3_feature_net,
                            self.w2_feature_net,
                            self.e2_discrete,
                            self.e2_feature_net,
                            self.add_cme_intercept,
                            self.add_m0_intercept)

    #fit thr trained model
    if split:
      dfa.fit_t(source_traindata2,
                source_traindata3,
                target_traindata,
                self.lam_set["cme"],
                self.lam_set["m0"])
    else:
      dfa.fit_t(source_traindata,
                source_traindata,
                target_traindata,
                self.lam_set["cme"],
                self.lam_set["m0"])
    return dfa


  def cme_w_x_feature_update(self,
                            target_traindata,
                            verbose=-1):
    """
    Estimate cme(w|x), update x feature kernel
    """

    if self.x1_target_feature_net:
      self.x1_target_feature_net.train(True)

    self.x2_feature_net.train(False)
    self.x3_feature_net.train(False)

    self.w2_feature_net.train(False)

    if not self.e2_discrete:
      self.e2_feature_net.train(False)

    with torch.no_grad():

      if target_traindata is not None:
        w2_target_feature = self.w2_feature_net(target_traindata.W)

    for _ in range(self.cme_iter):
      if self.x1_target_feature_net and target_traindata is not None:
        self.x1_target_opt.zero_grad()

        x1_target_feature = self.x1_target_feature_net(target_traindata.X)
        tmp2 = DeepMultiEnvGraph.augment_single_feature(x1_target_feature,
                                                   self.add_cme_intercept)
        x1_target_feature = tmp2
        loss = linear_reg_loss(w2_target_feature,
                             x1_target_feature,
                             self.lam_set["cme"])

        loss.backward()
        self.track_loss["w_x.x_target"].append(loss.item())


        if verbose >= 2:
          logger.info(f"cme_w_x x learning: {loss.item()}")

        if self.x1_target_feature_net:
          self.x1_target_opt.step()

  def cme_w_xe_feature_update(self, train_data, verbose):
    """
    Estimate cme(w|e,x), update e,x feature kernel
    """

    if self.x1_target_feature_net:
      self.x1_target_feature_net.train(False)


    self.x2_feature_net.train(True)
    self.x3_feature_net.train(False)

    self.w2_feature_net.train(False)
    if not self.e2_discrete:
      self.e2_feature_net.train(True)


    with torch.no_grad():
      w2_feature = self.w2_feature_net(train_data.W)

    for _ in range(self.cme_iter):
      self.x2_opt.zero_grad()
      if not self.e2_discrete:
        self.e2_opt.zero_grad()
        e2_feature = self.e2_feature_net(train_data.E)
      else:
        e2_feature = train_data.E

      x2_feature = self.x2_feature_net(train_data.X)

      if not self.e2_discrete:
        features = DeepMultiEnvGraph.augment_xe_feature(x2_feature,
                                                        e2_feature,
                                                        self.add_cme_intercept)



        loss = linear_reg_loss(w2_feature, features, self.lam_set["cme"])

      else:
        loss = torch.zeros(1)
        unique_e, unique_id = torch.unique(e2_feature,
                                         sorted=True,
                                         return_inverse=True)
        for _, i in enumerate(unique_e):
          con = unique_id==i
          select_id = con.nonzero()

          loss += linear_reg_loss(w2_feature[select_id,...],
                                  x2_feature[select_id,...],
                                  self.lam_set["cme"])

      loss.backward()
      self.track_loss["w_xe"].append(loss.item())
      if verbose >= 2:
        logger.info(f"cme_w_xe xe learning: {loss.item()}")
      self.x2_opt.step()
      if not self.e2_discrete:
        self.e2_opt.step()



  def m0_feature_update(self, train_data1, train_data2, verbose):
    """
    Estimate m0, cme(w|e,x), update w feature kernel, update m0
    """

    if self.x1_target_feature_net:
      self.x1_target_feature_net.train(False)


    self.x2_feature_net.train(False)
    self.x3_feature_net.train(True)

    self.w2_feature_net.train(True)
    if not self.e2_discrete:
      self.e2_feature_net.train(False)


    with torch.no_grad():
      if not self.e2_discrete:
        e2_feature1 = self.e2_feature_net(train_data1.E)
        e2_feature2 = self.e2_feature_net(train_data2.E)

      else:
        e2_feature1 = train_data1.E
        e2_feature2 = train_data2.E

      x2_feature1 = self.x2_feature_net(train_data1.X)
      x2_feature2 = self.x2_feature_net(train_data2.X)


    for _ in range(self.m0_iter):
      self.x3_opt.zero_grad()
      self.w2_opt.zero_grad()

      w2_feature1 = self.w2_feature_net(train_data1.W)
      x3_feature2 = self.x3_feature_net(train_data2.X)

      res = DeepMultiEnvGraph.fit_m0(x2_feature1,
                                     x2_feature2,
                                     x3_feature2,
                                     e2_feature1,
                                     e2_feature2,
                                     w2_feature1,
                                     train_data2.Y,
                                     self.lam_set["cme"],
                                     self.lam_set["m0"],
                                     self.add_cme_intercept,
                                     self.add_m0_intercept,
                                     self.e2_discrete)
      loss = res["loss"]
      loss.backward()# update Model2 parameters
      self.coef_m0 = res["alpha"]
      self.track_loss["m0"].append(loss.item())
      if verbose >= 2:
        logger.info(f"m0 learning: {loss.item()}")
      self.x3_opt.step()
      self.w2_opt.step()
    #return res["alpha"]


