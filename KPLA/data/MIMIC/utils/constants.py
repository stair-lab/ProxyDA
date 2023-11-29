import os

repo_path = os.path.abspath(os.path.join(__file__, "../.."))
print(repo_path)

PHYSIONET_DIR = "/shared/rsaas/oes2/mimic-cxr/physionet.org/"
EXPERIMENTS_DIR = os.path.join(repo_path, "experiments/")
RESULTS_DIR = os.path.join(repo_path, "results/")

METADATA_PATH = os.path.join(PHYSIONET_DIR, "mimic-cxr-2.0.0-metadata.csv")
CHEXPERT_PATH = os.path.join(PHYSIONET_DIR, "mimic-cxr-2.0.0-chexpert.csv")
ICD9_PATH = os.path.join(PHYSIONET_DIR, "diagnoses_icd.csv")
PATIENTS_PATH = os.path.join(PHYSIONET_DIR, "patients.csv")
ADMISSIONS_PATH = os.path.join(PHYSIONET_DIR, "admissions.csv")

CXR_EMB_PATH = "/shared/rsaas/oes2/physionet.org/mimic_cxr_embs.npz"
REPORT_EMB_PATH = "/shared/rsaas/oes2/physionet.org/mimic_report_embs.npz"

DICOM_SPLITS_DIR = "/shared/rsaas/oes2/physionet.org/dicom_data_splits/"

ALGORITHMS = ["ERM-SOURCE", "COVAR", "LABEL", "BBSE",
              "LSA-ORACLE", "LSA-ORACLE_CM_HARD", "LSA-ORACLE_CM_SOFT", "LSA-ORACLE_CM_EM",
              "ERM-TARGET", "True_w-LSA-ORACLE",
              "f_U_X", "f_Y_XU", "f_D_X", "f_U_X_tilde", "f_Y_XU_tilde",
              "Target_f_U_X", "Target_f_Y_XU",
              "WAE", "LSA-WAE-S"]

# TODO - [Wale] this is hacky; tried to make an Enum but not serializable to json.
LabelShiftApproaches = ["cm_hard", "cm_soft", "EM", 'true']