import os
import glob
import htcondor

jobs_dir = "/home/nschiou2/proxy_latent_shifts/tests/sim_multisource_v2/proposed_results"
os.makedirs(jobs_dir, exist_ok=True)

SCHEDD = htcondor.Schedd()

EXEC = "/home/nschiou2/miniconda3/envs/proxy/bin/python"
SCRIPT = "/home/nschiou2/proxy_latent_shifts/tests/sim_multisource_v2/test_proposed_onehot.py"

req = ""
req += '(Machine == "vision-21.cs.illinois.edu") || '
req += '(Machine == "vision-22.cs.illinois.edu") || '
req += '(Machine == "vision-23.cs.illinois.edu") '


params = {
    "var": 1.0,
    "mean": 0,
}

test_set = [params]
for par in test_set:
    for seed in [0.8, 0.6, 0.4, 0.2, 0.9, 0.7, 0.5, 0.3, 0.1]:

        mean = par["mean"]
        var = par["var"]

        run_name = f"test_proposed_v2_onehot_{seed}_m_{mean}_var_{var}"
        fixs = 0
        if fixs <= 1:
            run_name = run_name + "_fixscale"

        out_dir = f"/home/nschiou2/proxy_latent_shifts/tests/sim_multisource_v2/proposed_results/{run_name}"
        if len(glob.glob(os.path.join(out_dir, "*.csv"))) > 0:
            print(f"Skipping {run_name}.")
            continue

        arguments = [
            f"{SCRIPT}",
            "--s",
            str(seed),
            "--var",
            str(par["var"]),
            "--mean",
            str(par["mean"]),
            "--fixs",
            str(fixs),
            "--fname",
            run_name,
        ]

        job = {
            "executable": EXEC,
            "arguments": " ".join(arguments),
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
