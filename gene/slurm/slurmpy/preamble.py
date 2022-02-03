"""
Created by Ben Sepanski
"""

from machineconfig import MachineConfig


def build_slurm_gpu_preamble(config, num_gpus, job_name,
                             num_nodes=None,
                             num_cpus_per_task=None,
                             email_address=None,
                             account_name=None,
                             time_limit=None):
    if not isinstance(config, MachineConfig):
        raise TypeError(f"Expected config to be of type {MachineConfig.__class__.__name__}, not {type(config)}")

    min_num_nodes = (num_gpus + config.gpus_per_node - 1) // config.gpus_per_node
    if num_nodes is None:
        num_nodes = min_num_nodes
    if num_nodes < min_num_nodes:
        raise ValueError(f"Need at least {min_num_nodes} nodes for {num_gpus} gpus on {config.name}")
    num_mpi_tasks = num_gpus
    max_num_cpus_per_task = config.sockets_per_node \
                            * (config.cores_per_socket // num_mpi_tasks) \
                            * config.threads_per_core
    if num_cpus_per_task is None:
        num_cpus_per_task = max_num_cpus_per_task
    if num_cpus_per_task > max_num_cpus_per_task:
        raise ValueError(f"At most {max_num_cpus_per_task} are allowed per task with {num_nodes} nodes on {config.name}")

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
#SBATCH --mail-type=FAIL"""
    return preamble
