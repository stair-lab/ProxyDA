import os
import glob
import itertools
import htcondor

############### EDIT ############################
CODE_DIR = "/home/nschiou2/proxy_latent_shifts/"
EXEC = "/home/nschiou2/miniconda3/envs/proxy/bin/python"
EXPT_DIR = "tests/sim_multisource_cont/"
############### EDIT ############################

jobs_dir = os.path.join(CODE_DIR, EXPT_DIR, "jobs_proposed")
os.makedirs(jobs_dir, exist_ok=True)

SCHEDD = htcondor.Schedd()

SCRIPT = os.path.join(CODE_DIR, EXPT_DIR, "test_proposed_onehot.py")
OUT_DIR = os.path.join(CODE_DIR, EXPT_DIR, "model_select")

req = ""
req += '(Machine == "vision-21.cs.illinois.edu") || '
req += '(Machine == "vision-22.cs.illinois.edu") || '
req += '(Machine == "vision-23.cs.illinois.edu") '


params = {
    "var": 1.0,
    "mean": 0.0,
    "fixs": 1.0,
}
a_list = [1, 2, 3, 4, 5]
b_list = [1, 2, 3, 4, 5]

test_set = [params]
for par in test_set:
    for a, b in itertools.product(a_list, b_list):

        mean = par["mean"]
        var = par["var"]
        fixs = par["fixs"]

        run_name = f"test_proposed_onehot_a{a}b{b}"
        if fixs:
            run_name = run_name + "_fixscale"

        if len(glob.glob(os.path.join(OUT_DIR, run_name, "*.csv"))) > 0:
            print(f"Skipping {run_name}.")
            continue

        arguments = [
            f"{SCRIPT}",
            "--a",
            str(a),
            "--b",
            str(b),
            "--var",
            str(par["var"]),
            "--mean",
            str(par["mean"]),
            "--fixs",
            str(fixs),
            "--fname",
            run_name,
            "--outdir",
            OUT_DIR,
        ]

        job = {
            "executable": EXEC,
            "arguments": " ".join(arguments),
            "getenv": True,
            # Reqs
            "request_cpus": "16",
            "request_gpus": "1",
            "requirements": req,
            # Output files
            "output": os.path.join(jobs_dir, f"{run_name}.out"),
            "error": os.path.join(jobs_dir, f"{run_name}.err"),
            "log": os.path.join(jobs_dir, f"{run_name}.log"),
        }

        sub = htcondor.Submit(job)
        SCHEDD.submit(sub)

        print(f"Submitting {run_name}.")
