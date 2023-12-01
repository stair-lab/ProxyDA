"""MIMIC dataloader."""

import datetime as dt

import numpy as np
import pandas as pd
#import tensorflow as tf
from KPLA.data.MIMIC.utils.constants import (ADMISSIONS_PATH, CHEXPERT_PATH, ICD9_PATH,
                                  METADATA_PATH, PATIENTS_PATH)
import KPLA.data.MIMIC.utils.file_io as io


def load_data(metadata_path: str = METADATA_PATH,
            chexpert_path: str = CHEXPERT_PATH,
            icd9_path: str = ICD9_PATH,
            patients_path: str = PATIENTS_PATH,
            admissions_path: str = ADMISSIONS_PATH,
            drop_duplicate_subjects: bool = False,
            drop_duplicate_dicoms: bool = True):
  """Load data from csv across different linked mimic versions
  Input:
      metadata_path:    mimic-cxr-2.0.0-metadata.csv
      chexpert_path:    mimic-cxr-2.0.0-chexpert.csv
      icd9_path:        diagnoses_icd.csv
      patients_path:    patients.csv
      admissions_path:  admissions.csv
  Return:
      df:  pandas DataFrame combining all data and doing any preprocessing
  """
  AGE_BINS = [18, 40, 60, 80]

  # Load data .csv files
  metadata = io.callable_read(
      pd.read_csv, filepath_or_buffer=metadata_path
  ).drop_duplicates(subset=["subject_id"])
  patients = io.callable_read(
      pd.read_csv, filepath_or_buffer=patients_path)
  chexpert = io.callable_read(
      pd.read_csv, filepath_or_buffer=chexpert_path
  ).fillna(0.).astype(int)
  admissions = io.callable_read(
      pd.read_csv, filepath_or_buffer=admissions_path)

  CONDITIONS = list(chexpert.columns)[2:]

  # Combining data
  combined_data = metadata.merge(
      chexpert, how="inner", on=["subject_id", "study_id"])
  combined_data = combined_data.merge(
      admissions, how="inner", on="subject_id")
  combined_data = combined_data.merge(patients, how="inner", on="subject_id")

  # Preprocessing
  combined_data.admittime = combined_data.admittime.apply(
      lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))
  combined_data.dischtime = combined_data.dischtime.apply(
      lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"))

  # Remove examples labelled as uncertain
  for condition in CONDITIONS:
    combined_data = combined_data[combined_data[condition] != -1]

  # Assign U variable
  combined_data["white"] = combined_data.race.str.contains(
      "white", case=False).astype(int)

  conditions = [
      (combined_data.loc[:, "anchor_age"] >= AGE_BINS[0]) &
      (combined_data.loc[:, "anchor_age"] < AGE_BINS[1]),
      (combined_data.loc[:, "anchor_age"] >= AGE_BINS[1]) &
      (combined_data.loc[:, "anchor_age"] < AGE_BINS[2]),
      (combined_data.loc[:, "anchor_age"] >= AGE_BINS[2]) &
      (combined_data.loc[:, "anchor_age"] < AGE_BINS[3]),
      (combined_data.loc[:, "anchor_age"] >= AGE_BINS[3])
  ]
  values = ["18-39", "40-59", "60-79", "80+"]  # Age range for each age bin
  combined_data["age_bin"] = np.select(conditions, values)
  combined_data["age_bin_label"] = np.select(conditions, [0, 1, 2, 3])
  combined_data = combined_data.assign(
      age_old=lambda x: (
          x.anchor_age > combined_data.anchor_age.median()
      ).astype(int))

  # Intersectional $U$ splitting conditions
  conditions = [
      (combined_data.age_bin == "18-39") & (
          (combined_data["Cardiomegaly"] == 0) &
          (combined_data["Atelectasis"] == 0) &
          (combined_data["Pleural Effusion"] == 0) &
          (combined_data["Support Devices"] == 0)),
      (combined_data.age_bin == "18-39") & (
          (combined_data["Cardiomegaly"] == 1) |
          (combined_data["Atelectasis"] == 1) |
          (combined_data["Pleural Effusion"] == 1) |
          (combined_data["Support Devices"] == 1)),
      (combined_data.age_bin == "80+") & (
          (combined_data["Cardiomegaly"] == 0) &
          (combined_data["Atelectasis"] == 0) &
          (combined_data["Pleural Effusion"] == 0) &
          (combined_data["Support Devices"] == 0)),
      (combined_data.age_bin == "80+") & (
          (combined_data["Cardiomegaly"] == 1) |
          (combined_data["Atelectasis"] == 1) |
          (combined_data["Pleural Effusion"] == 1) |
          (combined_data["Support Devices"] == 1)),
  ]
  combined_data["age_bin_condition"] = np.select(conditions, [0, 1, 2, 3])

  conditions = [
      (combined_data.age_old == 0) & (
          (combined_data["Cardiomegaly"] == 0) &
          (combined_data["Atelectasis"] == 0) &
          (combined_data["Pleural Effusion"] == 0)),
      (combined_data.age_old == 0) & (
          (combined_data["Cardiomegaly"] == 1) |
          (combined_data["Atelectasis"] == 1) |
          (combined_data["Pleural Effusion"] == 1)),
      (combined_data.age_old == 1) & (
          (combined_data["Cardiomegaly"] == 0) &
          (combined_data["Atelectasis"] == 0) &
          (combined_data["Pleural Effusion"] == 0)),
      (combined_data.age_old == 1) & (
          (combined_data["Cardiomegaly"] == 1) |
          (combined_data["Atelectasis"] == 1) |
          (combined_data["Pleural Effusion"] == 1)),
  ]
  combined_data["age_old_condition"] = np.select(conditions, [0, 1, 2, 3])

  conditions = [
      ((combined_data["Cardiomegaly"] == 0) &
        (combined_data["Atelectasis"] == 0) &
        (combined_data["Pleural Effusion"] == 0)),
      ((combined_data["Cardiomegaly"] == 1) |
        (combined_data["Atelectasis"] == 1) |
        (combined_data["Pleural Effusion"] == 1)),
  ]
  combined_data["sick"] = np.select(conditions, [0, 1])

  combined_data.index = combined_data.dicom_id
  if drop_duplicate_subjects:
    combined_data = combined_data.drop_duplicates(subset=["subject_id"])
  if drop_duplicate_dicoms:
    combined_data = combined_data.drop_duplicates(subset=["dicom_id"])

  return combined_data


def convert_data_Y2D(source, target):
  source = list(source)
  target = list(target)

  source[1] = np.zeros_like(source[1])
  target[1] = np.ones_like(target[1])

  domain_D = tuple(
      np.concatenate([s, t], 0) for s, t in zip(source, target)
      )

  return domain_D

"""
def convert_data_X2XU(D, u_dim):
    D = list(D)
    x, u = D[0], D[4]
    u = tf.cast(u, tf.int64)
    u = tf.one_hot(indices=tf.reshape(u, [-1]),
                   depth=u_dim).numpy()
    D[0] = np.concatenate([x, u], axis=1)

    return tuple(D)
"""

def generate_noisy_u(labels, num_classes=4, p=0.25):
  p = p ** (1 / (num_classes - 1))

  # Assuming your array of labels is called 'labels'
  num_labels = len(labels)

  # Create a probability matrix with shape (num_labels, num_classes)
  prob_matrix = np.zeros((num_labels, num_classes))
  for i in range(num_labels):
      for j in range(num_classes):
          if labels[i] != j:
              prob_matrix[i, j] = p

  # Sample a binary matrix with shape (num_labels, num_classes) using the
  # probability matrix
  flip_mask = np.random.binomial(n=1, p=prob_matrix)

  # Flip the labels according to the binary matrix
  flipped_labels = np.zeros(num_labels, dtype=int)
  for i in range(num_labels):
    if np.sum(flip_mask[i]) < num_classes-1:
      flipped_labels[i] = labels[i]
    else:
      flipped_labels[i] = np.random.choice(
        np.where(flip_mask[i] == 1)[0])

  return flipped_labels.reshape(labels.shape)


def convert_data_tuple2dict(d):
  vars_ = ["x", "y", "c", "w", "u"]
  return {k: v for k, v in zip(vars_, d)}