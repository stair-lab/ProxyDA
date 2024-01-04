import os
import htcondor

jobs_dir = "/home/kt14/workbrench/backup/proxy_latent_shifts/tests/mimic_w_concepts/experiments/baseline_jobs"
os.makedirs(jobs_dir, exist_ok=True)

SCHEDD = htcondor.Schedd()

EXEC = '/home/oes2/miniconda3/envs/lsa/bin/python'
SCRIPT = "/home/oes2/proxy_latent_shifts/tests/mimic_w_concepts/experiments/test_baseline.py"

for seed in range(192, 202):
    run_name = f"test_baseline_MIMIC_{seed}"
    arguments = [ 
    f"{SCRIPT}",
        '--seed', str(seed),
        '--output_dir', f"/home/oes2/proxy_latent_shifts/tests/mimic_w_concepts/experiments/results_svm_dim_red/{run_name}",
        '--n_params', '7',
        '--n_folds', '3',
        '--reduce', '64'
    ]

    job = {
        "executable": EXEC,
        "arguments": ' '.join(arguments),

        "batch-name": run_name,

        # reqs
        "request_cpus": '16',
        "request_gpus": '1',
        "requirements": '(CUDADeviceName != "NVIDIA GeoForce GTX TITAN X") ',

        # output files
        "output": os.path.join(jobs_dir, f'{run_name}.out'),
        "error": os.path.join(jobs_dir, f'{run_name}.err'),
        "log": os.path.join(jobs_dir, f'{run_name}.log'),
    }

    sub = htcondor.Submit(job)
    SCHEDD.submit(sub)

    print(f"Submitting {run_name}. {job}")