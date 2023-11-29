import json
import os
import shutil
import time
import pickle


def makedirs(path, exist_ok=True, change_group=True, group="vision_fmri",
            max_retry=20):
  success = False
  for i in range(max_retry):
    try:
      os.makedirs(path, exist_ok=exist_ok)
      if change_group:
        chgrp(path, group=group)
      success = True
      break
    except OSError:
      time.sleep(i)
  if not success:
    raise OSError(f"Failed to create directory {path}.")


def chgrp_dir(dir, group="vision-fmri", max_retry=20):
  success = False
  for root, dirs, files in os.walk(dir):
    for i in range(max_retry):
      try:
        shutil.chown(root, group=group)
        for f in files:
            shutil.chown(os.path.join(root, f), group=group)
        success = True
        break
      except OSError:
        time.sleep(i)
  if not success:
    raise OSError(f"Failed to change group of {dir}.")


def chgrp(path, group="vision_fmri", max_retry=20):
  success = False
  for i in range(max_retry):
    try:
      shutil.chown(path, group=group)
      success = True
      break
    except OSError:
      time.sleep(i)
  if not success:
    raise OSError(f"Failed to change group of {path}.")


def write_json_to_drive(path, data, mode="w", change_group=True,
                        group="vision_fmri", max_retry=20):
  success = False
  for i in range(max_retry):
    try:
      with open(path, mode) as f:
        json.dump(data, f, indent=2)
      if change_group:
        chgrp(path, group=group)
      success = True
      break
    except OSError:
      time.sleep(i)
  if not success:
    raise OSError(f"Failed to write .json file to {path}.")


def write_done_to_drive(path, mode="w", change_group=True,
                        group="vision_fmri", max_retry=20):
  success = False
  for i in range(max_retry):
    try:
      with open(os.path.join(path, "done"), mode=mode):
        time.sleep(1)
      if change_group:
        chgrp(os.path.join(path, "done"), group=group)
      success = True
      break
    except OSError:
      time.sleep(i)
  if not success:
    raise OSError(f"Failed to write `done` to {path}.")


def callable_read(fn, max_retry=20, *args, **kwargs):
  """Passes an arbitrary callable that reads from /shared/rsaas/."""
  success = False
  for i in range(max_retry):
    try:
      result = fn(*args, **kwargs)
      success = True
      break
    except OSError:
      time.sleep(i)
  if not success:
    raise OSError("Failed to read from /shared/rsaas/.")
  return result


def read_dicom_splits(path, max_retry=20):
  """Read DICOM splits from /shared/rsaas."""
  success = False
  for i in range(max_retry):
    try:
      with open(path, "rb") as f:
        dicom_splits = pickle.load(f)["domain_splits"]
      success = True
      break
    except OSError:
      time.sleep(i)
  if not success:
    raise OSError("Failed to read from /shared/rsaas/.")
  return dicom_splits