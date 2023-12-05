import random
from multiprocessing import Pool
from typing import List

import numpy as np
import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm

import KPLA.data.MIMIC.utils.file_io as io


class MIMIC(tf.keras.utils.Sequence):
  def __init__(self, dicom_ids: list,
              metadata: pd.DataFrame, cxr_emb_path: str,
              report_emb_path: str,
              Y=["No Finding"], U=["white"], W=["race"],
              shuffle: bool = True,
              batch_size: int = 1,
              verbose: int = 0
              ):
    """ Loader MIMIC-CXR Latent Shift Adaptation (X, Y, C, W, U)
    Input
        dicom_ids: list of mimic dicom ids. If empty, metadata dicom ids
            will be used
        metadata: a dataframe minimaly with dicom_id, participant_id and
            study_id
        cxr_emb_path: numpy npz with dicom_id and image embeddings
        report_emb_path: numpy npz with study_id and report embeddings
        Y: list of Y columns given a row in metadata with dicom_id and
            study_id
        W: list of W columns given a row in metadata with dicom_id and
            study_id
        U: list of U columns given a row in metadata with dicom_id and
            study_id
    """
    self.shuffle = shuffle
    self.batch_size = batch_size

    self.dicom_ids = dicom_ids

    self.cxr_emb = io.callable_read(np.load, file=cxr_emb_path)
    self.report_emb = io.callable_read(np.load, file=report_emb_path)

    if len(self.dicom_ids) == 0:
      self.dicom_ids = list(metadata.dicom_id)

    self.metadata = metadata[
        (metadata.study_id.isin(self.report_emb.keys())) &
        (metadata.dicom_id.isin(self.cxr_emb.keys())) &
        (metadata.dicom_id.isin(self.dicom_ids))
      ]

    for w in W:
      self.metadata.loc[:, 'orig_'+w] = self.metadata.loc[:, w].values
      self.metadata.loc[:, w] = pd.factorize(self.metadata.loc[:, w].values)[0]

    if self.metadata.shape[0] != len(dicom_ids) and verbose > 0:
      print("Not all dicom_ids were present. Size: {} -> {}".format(
            len(self.dicom_ids), self.metadata.shape[0]))

    self.dicom_ids = list(self.metadata.dicom_id)

    assert len(self.dicom_ids) == self.metadata.shape[0]

    self.U = U
    self.W = W
    self.Y = Y

  def load_cxr(self, k: str):
    """Load cxr embedding
    Input
        k: dicom_id string
    Return
        v: cxr embedding vector -- numpy array
    """
    return self.cxr_emb.get(k).reshape(1, -1)

  def load_report(self, k: str):
    """Load cxr embedding
    Input
        k: study_id string
    Return
        v: Bio_ClinialBERT embedding vector -- numpy array
    """
    return self.report_emb.get(k).reshape(1, -1)

  def parallel_read(self, func, files: List[str], verbose: bool=False, name: str=''):
    """Load embeddings in parallel
    Input
        func:       function to read embedding given a string
        files:      list of keys to pass to func
    Return
        embeddings: list of embedding vectors -- list[numpy arrays]
    """
    with Pool() as pool:
      x_list = list(tqdm(pool.map(func, files), total=len(files), desc=f"reading {name}"))
      pool.close()
    return x_list

  def serial_read(self, func, files: List[str], verbose: bool=False, name: str=''):
    """Load embeddings in series
    Input
        func:       function to read embedding given a string
        files:      list of keys to pass to func
    Return
        embeddings: list of embedding vectors -- list[numpy arrays]
    """
    if verbose:
      return list(tqdm([func(k) for k in files], total=len(files), desc=f"reading {name}"))
    else:
      return [func(k) for k in files]

  def __getitem__(self, index):
    dicom_ids = self.dicom_ids[
        index*self.batch_size:(index+1)*self.batch_size]
    metadata = self.metadata[self.metadata.dicom_id.isin(dicom_ids)]
    study_ids = list(metadata.study_id)

    Y = metadata[self.Y].values
    # W = metadata[self.W].values
    U = metadata[self.U].values
    W = np.random.normal(U)

    Xs = self.serial_read(self.load_cxr, dicom_ids)
    Cs = self.serial_read(self.load_report, study_ids)

    X = np.concatenate(Xs, 0)
    C = np.concatenate(Cs, 0)

    return X, Y, C, W, U

  def __len__(self):
    return int(np.floor(len(self.dicom_ids) / self.batch_size))

  def on_epoch_end(self, shuffle=True):
    if shuffle:
      random.shuffle(self.dicom_ids)

  def generate_data(self, parallel=False):
    """Generate data for all dicom_ids
    Return:
        data: tuple(X, Y, C, W, U)
    """
    dicom_ids = self.dicom_ids
    metadata = self.metadata[self.metadata.dicom_id.isin(dicom_ids)]
    study_ids = list(metadata.study_id)

    Y = metadata[self.Y].values
    # W = metadata[self.W].values
    U = metadata[self.U].values
    W = np.random.normal(U)

    reader = self.parallel_read if parallel else self.serial_read

    Xs = reader(self.load_cxr, dicom_ids, verbose=True, name='cxr')
    Cs = reader(self.load_report, study_ids, verbose=True, name='report')

    X = np.concatenate(Xs, 0)
    C = np.concatenate(Cs, 0)

    return X, Y, C, W, U