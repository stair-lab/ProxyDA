"""
Data class of the deep kernel method
"""

#Author: Katherine Tsai <kt14@illinois.edu>
#License: MIT

from typing import NamedTuple, Optional
import numpy as np
import torch

from sklearn.model_selection import train_test_split


class dfaDataSet(NamedTuple):
    C: np.ndarray
    X: np.ndarray
    W: np.ndarray
    Y: np.ndarray


class dfaDataSetTorch(NamedTuple):
    C: torch.Tensor
    X: torch.Tensor
    W: torch.Tensor
    Y: torch.Tensor
  
    @classmethod
    def from_numpy(cls, train_data: dfaDataSet):

        return dfaDataSetTorch(C=torch.tensor(np.asarray(train_data.C), dtype=torch.float32),
                                X=torch.tensor(np.asarray(train_data.X), dtype=torch.float32),
                                W=torch.tensor(np.asarray(train_data.W), dtype=torch.float32),
                                Y=torch.tensor(np.asarray(train_data.Y), dtype=torch.float32))

    def to_gpu(self):
        return dfaDataSetTorch(C=self.C.cuda(),
                                X=self.X.cuda(),
                                W=self.W.cuda(),
                                Y=self.Y.cuda())




def split_train_data(train_data: dfaDataSetTorch, n_split):

    n_data = train_data[0].shape[0]
    split_size = int(n_data/n_split)
    size_index = [split_size for i in range(n_split)]
    size_index[-1] = n_data - split_size*(n_split-1)

    split_index = torch.utils.data.random_split(range(n_data), size_index)

    def get_data(data, idx):
        return data[idx] if data is not None else None

    split_data = []
    for s_idx in split_index:
        idx = [i for i in s_idx]
        batch_data = dfaDataSetTorch(*[get_data(data, idx) for data in train_data])
        split_data.append(batch_data)
        
    return split_data