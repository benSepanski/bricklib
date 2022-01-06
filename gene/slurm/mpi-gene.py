import argparse
import os


class MachineConfig:
    def __init__(self, 
                 gpus_per_node=None,
                 sockets_per_node=None,
                 cores_per_socket=None,
                 threads_per_core=None,
                 ):
        self.gpus_per_node = gpus_per_node
        self.sockets_per_node = sockets_per_node
        self.cores_per_socket = cores_per_socket
        self.threads_per_core = threads_per_core


machine_configurations = {
    "perlmutter": MachineConfig(gpus_per_node=4,
                                sockets_per_node=1,
                                cores_per_socket=64,
                                threads_per_core=2,
                                ),
}


def build_slurm_preamble(config, num_gpus, job_name, email_address=None, account_name=None, time_limit=None):
    num_nodes = (num_gpus + config.gpus_per_node - 1) // config.gpus_per_node
    num_mpi_tasks = num_gpus
    num_cpus_per_task = config.sockets_per_node \
        * (config.cores_per_socket // num_mpi_tasks) \
        * config.threads_per_core

    preamble = f"""#!/bin/bash
#SBATCH -C gpu
#SBATCH -N {num_nodes}
#SBATCH -G {num_gpus}
#SBATCH -n {num_mpi_tasks}
#SBATCH -c {num_cpus_per_task}
#SBATCH --gpus-per-task 1
#SBATCH --gpu-bind single:1
#SBATCH -q regular
#SBATCH -J {job_name}
#SBATCH -o {job_name}.out
#SBATCH -e {job_name}.err
#SBATCH -t {time_limit}"""
    if account_name is not None:
        preamble += f"\n#SBATCH -A {account_name}"
    if email_address is not None:
        preamble += f"""\n#SBATCH --mail-user={email_address}
#SBATCH --mail-type=ALL"""
    return preamble


if __name__ == "__main__":
    known_machines = [machine for machine in machine_configurations]
    parser = argparse.ArgumentParser("Build slurm job file for MPI Gene")
    parser.add_argument("gpus", metavar="G", type=int, help="Number of GPUs")
    parser.add_argument("-J", "--job_name", type=str, help="Job name", default=None)
    parser.add_argument('-M', "--machine", type=str, help=f"Machine name, one of {known_machines}")
    parser.add_argument("-e", "--email", type=str, help="Email address to notify on job completion")
    parser.add_argument("-A", "--account", type=str, help="slurm account to charge")
    parser.add_argument("-t", "--time-limit", type=str, help="Time limit, in format for slurm", default="01:00:00")
    parser.add_argument("--bricks-build-dir", type=str, help="Build directory to use for bricks", default=".")
    parser.add_argument("--compiler", type=str, help="Compiler being used for affinity check", default="gnu")
    parser.add_argument("-d", "--per-process-domain-size", type=str, 
                        help="Per-process domain extent formatted as I,J,K,L,M,N",
                        default="72,32,24,24,32,2")

    args = vars(parser.parse_args())
    
    machine_name = args["machine"]
    if machine_name not in machine_configurations:
        raise ValueError(f"Unrecognized machine {machine_name}, must be one of {known_machines}")
    machine_config = machine_configurations[machine_name]
    num_gpus = args["gpus"]
    job_name = args["job_name"]
    if job_name is None:
        job_name = "mpi-gene-" + machine_name + f"-{num_gpus}gpus"

    time_limit = args["time_limit"]
    account_name = args["account"]
    email_address = args["email"]
    compiler = args["compiler"]

    preamble = build_slurm_preamble(machine_config, num_gpus, job_name,
                                    email_address=email_address,
                                    account_name=account_name,
                                    time_limit=time_limit)

    gtensor_dir = os.getenv("gtensor_DIR")
    if gtensor_dir is None:
        raise ValueError("Environment variable gtensor_DIR must be set")
    bricks_dir = args["bricks_build_dir"]

    environment_setup = f"""#https://docs.nersc.gov/jobs/affinity/#openmp-environment-variables
export OMP_PLACES=threads
export OMP_PROC_BIND=true
export gtensor_DIR={gtensor_dir}
export bricks_DIR={bricks_dir}"""

    run_jobs = ""

    def add_job(per_process_extent, procs_per_dim):
        global run_jobs
        srun_cmd = "srun --cpu-bind=cores"
        run_jobs += f"\n{srun_cmd} ${{bricks_DIR}}/gene/mpi-gene6d \
    -d {','.join(map(str,per_process_extent))} \
    -p {','.join(map(str,procs_per_dim))} \
    -I 100 -W 5 -a"

    per_process_extent = tuple(map(int, args["per_process_domain_size"].split(',')))
    
    # First do the strong scaling jobs
    run_jobs += "\n# strong scaling jobs"
    for k in range(1, num_gpus+1):
        if per_process_extent[2] % k == 0:
            for ell in range(1, num_gpus+1):
                if k * ell <= num_gpus and per_process_extent[3] % ell == 0:
                    for m in range(1, num_gpus+1):
                        if k * ell * m == num_gpus and per_process_extent[4] % m == 0:
                            procs_per_dim = [1, 1, k, ell, m, 1]
                            divided_extent = [p for p in per_process_extent]
                            for index, divisor in [(2, k), (3, ell), (4, m)]:
                                divided_extent[index] //= divisor
                            add_job(divided_extent, procs_per_dim)

    # Next do the weak scaling jobs
    run_jobs += "\n\n# weak scaling jobs"
    for k in range(1, num_gpus+1):
        for ell in range(1, num_gpus+1):
            if k * ell <= num_gpus:
                for m in range(1, num_gpus+1):
                    if k * ell * m == num_gpus:
                        procs_per_dim = [1, 1, k, ell, m, 1]
                        add_job(per_process_extent, procs_per_dim)

    slurm_script = "\n\n".join([preamble, environment_setup, run_jobs]) + "\n"

    print(slurm_script)
