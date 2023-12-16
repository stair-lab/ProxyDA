import os
import glob
import htcondor

jobs_dir = "/home/kt14/workbench/backup/proxy_latent_shifts/tests/mimic_w_concepts/experiments/jobs_proposed"
os.makedirs(jobs_dir, exist_ok=True)

SCHEDD = htcondor.Schedd()

EXEC = '/home/kt14/miniconda3/envs/work2/bin/python'
SCRIPT = "/home/kt14/workbench/backup/proxy_latent_shifts/tests/mimic_w_concepts/experiments/test_proposed_v3.py"

req = ''
req += '(Machine == "vision-21.cs.illinois.edu") || '
req += '(Machine == "vision-22.cs.illinois.edu") || '
req += '(Machine == "vision-23.cs.illinois.edu") '

for seed in range(192,202):
    run_name = f"test_proposed_MIMIC_{seed}_multihead_cvacc"
    out_dir = f"/home/kt14/workbench/backup/proxy_latent_shifts/tests/mimic_w_concepts/experiments/proposed_results/{run_name}"
    if len(glob.glob(os.path.join(out_dir, '*.csv'))) > 0:
        print(f"Skipping {run_name}.")
        continue

    arguments = [ 
    f"{SCRIPT}",
        '--seed', str(seed),
        '--output_dir', out_dir,
        '--n_params', '3',
        '--n_folds', '3',
        '--reduce', '64',
    ]

    job = {
        "executable": EXEC,
        "arguments": ' '.join(arguments),

        # reqs
        "request_cpus": '16',
        "request_gpus": '1',
        "requirements": req,

        # output files
        "output": os.path.join(jobs_dir, f'{run_name}.out'),
        "error": os.path.join(jobs_dir, f'{run_name}.err'),
        "log": os.path.join(jobs_dir, f'{run_name}.log'),
    }

    sub = htcondor.Submit(job)
    SCHEDD.submit(sub)

    print(f"Submitting {run_name}.")
    # break