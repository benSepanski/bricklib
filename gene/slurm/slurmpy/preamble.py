"""
Created by Ben Sepanski
"""

from machineconfig import MachineConfig
from mailtypes import MailType
from typing import Union, Iterable


def build_slurm_gpu_preamble(config, num_gpus, job_name,
                             num_nodes: int = None,
                             num_cpus_per_task: int = None,
                             email_address: str = None,
                             account_name: str = None,
                             time_limit: str = None,
                             mail_type: Union[MailType, Iterable[MailType]] = None,
                             output_file_name: str = None,
                             error_file_name: str = None,
                             image_name: str = None,
                             ):
    if not isinstance(config, MachineConfig):
        raise TypeError(f"Expected config to be of type {MachineConfig.__class__.__name__}, not {type(config)}")

    if mail_type is None:
        mail_type = MailType.FAIL
    if isinstance(mail_type, MailType):
        mail_type = [mail_type]
    mail_type = tuple(mail_type)

    if output_file_name is None:
        output_file_name = f"{job_name}.out"
    if error_file_name is None:
        error_file_name = f"{job_name}.err"

    min_num_nodes = (num_gpus + config.gpus_per_node - 1) // config.gpus_per_node
    if num_nodes is None:
        num_nodes = min_num_nodes
    if num_nodes < min_num_nodes:
        raise ValueError(f"Need at least {min_num_nodes} nodes for {num_gpus} gpus on {config.name}")
    num_mpi_tasks = num_gpus
    max_num_cpus_per_node = config.sockets_per_node \
        * config.cores_per_socket \
        * config.threads_per_core
    max_num_tasks_per_node = (num_mpi_tasks + num_nodes - 1) // num_nodes
    max_num_cpus_per_task = max_num_cpus_per_node // max_num_tasks_per_node

    if num_cpus_per_task is None:
        num_cpus_per_task = max_num_cpus_per_task
    if num_cpus_per_task > max_num_cpus_per_task:
        raise ValueError(f"At most {max_num_cpus_per_task} are allowed per task with {num_nodes} nodes on {config.name}")

    preamble = f"""#!/bin/bash
#SBATCH -C gpu
#SBATCH -N {num_nodes}
#SBATCH -G {min(num_gpus, num_nodes * config.gpus_per_node)}
#SBATCH -n {num_mpi_tasks}
#SBATCH -c {num_cpus_per_task}
#SBATCH --gpus-per-node {min(num_gpus, config.gpus_per_node)}
#SBATCH -q regular
#SBATCH -J {job_name}
#SBATCH -o {output_file_name}
#SBATCH -e {error_file_name}
#SBATCH -t {time_limit}"""
    if account_name is not None:
        preamble += f"\n#SBATCH -A {account_name}"
    if email_address is not None:
        preamble += f"""\n#SBATCH --mail-user={email_address}
#SBATCH --mail-type={','.join(map(lambda m: m.name, mail_type))}"""
    if image_name is not None:
        preamble += f"\n#SBATCH --image={image_name}"
    return preamble
