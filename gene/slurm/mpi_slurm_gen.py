import argparse
import os
import stat
from typing import List, Tuple

import math

from slurmpy import *

if __name__ == "__main__":
    known_machines = [machine for machine in machine_configurations]
    parser = argparse.ArgumentParser("Build slurm job file for MPI Gene")
    parser.add_argument("gpus", metavar="G", type=int, help="Number of GPUs")
    parser.add_argument("-B", "--brick_shape", metavar="BI,BJ,BK,BL,BM,BN", type=str, help="Brick shape",
                        default="2,16,2,2,1,1")
    parser.add_argument('-M', "--machine", type=str, help=f"Machine name, one of {known_machines}")
    parser.add_argument("-e", "--email", type=str, help="Email address to notify on job completion")
    parser.add_argument("-A", "--account", type=str, help="slurm account to charge")
    parser.add_argument("-t", "--time-limit", type=str, help="Time limit, in format for slurm", default="01:00:00")
    parser.add_argument("--compiler", type=str, help="Compiler being used for affinity check", default="gnu")
    parser.add_argument("-w", "--weak", action='store_true', help="Perform weak scaling instead of strong scaling")
    parser.add_argument("--always-cuda-aware", action='store_true', help="Only use jobs which use CUDA-Aware MPI")
    parser.add_argument("-d", "--per-process-domain-size", type=str,
                        help="Per-process domain extent formatted as I,J,K,L,M,N",
                        default=None)
    parser.add_argument("--num-gz", type=int, help="Number of ghost-zones to use", default=1)
    parser.add_argument("-J", "--job-name", type=str, help="Name of job", default=None)
    parser.add_argument("-o", "--output_file", type=str, help="Output file to write to", default=None)

    args = vars(parser.parse_args())

    num_gpus = args["gpus"]
    brick_shape = tuple(map(int, args["brick_shape"].split(",")))
    assert len(brick_shape) == 6
    assert all([b > 0 for b in brick_shape])
    machine_name = args["machine"]
    if machine_name not in machine_configurations:
        raise ValueError(f"Unrecognized machine {machine_name}, must be one of {known_machines}")
    machine_config = machine_configurations[machine_name]

    email_address = args["email"]
    account_name = args["account"]
    time_limit = args["time_limit"]
    compiler = args["compiler"]

    weak_scaling: bool = args["weak"]
    always_cuda_aware: bool = args["always_cuda_aware"]

    if args["per_process_domain_size"] is not None:
        per_process_extent = tuple(map(int, args["per_process_domain_size"].split(',')))
    elif weak_scaling:
        per_process_extent = (72, 32, 32, 32, 32, 2)
    else:
        per_process_extent = (72, 32, 64, 64, 32, 2)
    num_gz = args["num_gz"]

    job_name = args["job_name"]
    if job_name is None:
        job_name = f"mpi_{num_gpus}{'_weak' if weak_scaling else ''}"
    output_file = args["output_file"]
    if output_file is None:
        output_file = f"{job_name}.csv"

    vec_len = 1
    vec_shape = [None for _ in range(len(brick_shape))]
    for i in range(len(brick_shape)):
        vec_shape[i] = max(min(brick_shape[i], 32 // vec_len), 1)
        vec_len *= vec_shape[i]
    assert vec_len == 32

    gtensor_dir = os.getenv("gtensor_DIR")
    if gtensor_dir is None:
        raise ValueError("Environment variable gtensor_DIR must be set")

    build_dir = os.path.abspath(f"cmake-builds/mpi/{machine_config.name}_brick_{'_'.join(map(str, brick_shape))}")
    executable_name = "gene/mpi-gene6d"

    def get_build_dir(cuda_aware: Union[bool, str]) -> str:
        if isinstance(cuda_aware, bool):
            cuda_aware = "ON" if cuda_aware else "OFF"
        return os.path.abspath(f"{build_dir}/cuda_aware_{cuda_aware}")


    def build_job(brick_dim, vec_dim, cuda_aware: bool):
        current_build_dir = get_build_dir(cuda_aware)
        return f"""
if [[ ! -d {current_build_dir} ]] ; then 
    mkdir -p {current_build_dir}
    echo "Building brick-dim {brick_dim} with vec dim {vec_dim}, cuda_aware={cuda_aware}" ; 
    cmake -S {os.path.abspath(os.path.join(os.path.pardir, os.path.pardir))} \\
        -B {current_build_dir} \\
        -DCMAKE_CUDA_ARCHITECTURES={machine_config.cuda_arch} \\
        -DCMAKE_INSTALL_PREFIX=bin \\
        -DGENE6D_USE_TYPES=OFF \\
        -DGENE6D_CUDA_AWARE={"ON" if cuda_aware else "OFF"} \\
        -DGENE6D_BRICK_DIM={','.join(map(str, reversed(brick_dim)))} \\
        -DGENE6D_VEC_DIM={','.join(map(str, reversed(vec_dim)))} \\
        -DCMAKE_CUDA_FLAGS=\"--resource-usage -lineinfo -gencode arch=compute_{machine_config.cuda_arch},""" \
               f"""code=[sm_{machine_config.cuda_arch},lto_{machine_config.cuda_arch}]\" \\
        -DCMAKE_BUILD_TYPE=Release \\
        -DPERLMUTTER={"ON" if machine_config.name == "perlmutter" else "OFF"} \\
    || exit 1
    (cd {current_build_dir} && make clean && make -j 20 mpi-gene6d || exit 1)
else
    # Someone else is building, wait for that to finish
    let max_count=20;
    let count=0;
    while [[ ! -f {current_build_dir}/{executable_name} ]] && [[ count -lt max_count ]]
    do
       echo "Waiting for some other process to finish partially complete build ${{count}} / ${{max_count}}"
       sleep 1
       let count=count+1;
    done
    if [[ ! -f {current_build_dir}/{executable_name} ]] ; then
        echo "Build failed!" ;
        exit 1 ;
    else
        echo "Build complete!" ;
    fi
fi
"""


    environment_setup = f"""module unload darshan
#https://docs.nersc.gov/jobs/affinity/#openmp-environment-variables
export OMP_PLACES=threads
export OMP_PROC_BIND=true
export gtensor_DIR={gtensor_dir}
export MPICH_RANK_REORDER_METHOD=3 #https://docs.nersc.gov/jobs/best-practices/#grid_order
"""


    class JobDescription:
        """
        A description of a job to run
        """

        # noinspection PyShadowingNames
        def __init__(self, per_process_extent, procs_per_dim, cuda_aware: bool):
            global num_gz
            global output_file
            self.per_process_extent = tuple(per_process_extent)
            self.procs_per_dim = tuple(procs_per_dim)
            self.num_gz = num_gz
            self.output_file = output_file
            self.cuda_aware = cuda_aware

        def get_node_local_procs_per_dim(self, config: MachineConfig) -> Tuple[int, int, int, int, int, int]:
            possible_k = list(filter(lambda k: self.procs_per_dim[2] % k == 0 and config.gpus_per_node % k == 0,
                                     range(1, min(self.procs_per_dim[2], config.gpus_per_node) + 1)))
            possible_ell = list(filter(lambda ell: self.procs_per_dim[3] % ell == 0 and config.gpus_per_node % ell == 0,
                                       range(1, min(self.procs_per_dim[3], config.gpus_per_node) + 1)))
            k_times_ell = min(config.gpus_per_node, self.procs_per_dim[2] * self.procs_per_dim[3])
            possible_k_ell = list(filter(lambda k_ell: k_ell[0] * k_ell[1] == k_times_ell,
                                         [(k, ell) for k in possible_k for ell in possible_ell]))
            assert len(possible_k_ell) > 0
            max_min_k_ell: int = max(map(lambda k_ell: min(*k_ell), possible_k_ell))
            best_k_ell: List[Tuple[int, int]] = list(filter(lambda k_ell: min(*k_ell) == max_min_k_ell, possible_k_ell))
            assert len(best_k_ell) > 0
            if len(best_k_ell) > 1:
                max_ell = max(map(lambda k_ell: k_ell[1], best_k_ell))
                best_k_ell = list(filter(lambda k_ell: k_ell[1] == max_ell, best_k_ell))
                assert len(best_k_ell) == 1
            k, ell = best_k_ell[0]
            assert k > 0, ell > 0
            assert config.gpus_per_node % (k * ell) == 0
            m = math.gcd(config.gpus_per_node // (k * ell), self.procs_per_dim[4])
            assert k * ell * m == config.gpus_per_node
            return 1, 1, k, ell, m, 1

        def __eq__(self, that):
            return self.per_process_extent == that.per_process_extent \
                   and self.procs_per_dim == that.procs_per_dim \
                   and self.num_gz == that.num_gz \
                   and self.cuda_aware == that.cuda_aware \

        def __ne__(self, that):
            return not (self == that)

        def __hash__(self):
            return hash((self.per_process_extent, self.procs_per_dim, self.num_gz))


    def run_jobs(cuda_aware: bool) -> List[JobDescription]:
        jobs_to_run = []
        ghost_depth = (0, 0, 2 * num_gz, 2 * num_gz, 0, 0)
        min_extent = tuple(map(lambda g: max(1, 3 * g), ghost_depth))
        seen_procs_per_dim = set()
        for m in filter(lambda m: per_process_extent[3] % m == 0 and num_gpus % m == 0, range(1, num_gpus+1)):
            for k in filter(lambda k: num_gpus % (k*m) == 0, range(1, num_gpus + 1)):
                ell = num_gpus // (k * m)
                assert m * k * ell == num_gpus
                procs_per_dim = (1, 1, k, ell, m, 1)
                if procs_per_dim in seen_procs_per_dim:
                    continue
                else:
                    seen_procs_per_dim.add(tuple(procs_per_dim))
                    assert procs_per_dim in seen_procs_per_dim
                if weak_scaling:
                    # weak scaling jobs:
                    job = JobDescription(per_process_extent=per_process_extent, procs_per_dim=procs_per_dim,
                                         cuda_aware=cuda_aware)
                    jobs_to_run.append(job)
                elif all([extent % num_procs == 0 for extent, num_procs in zip(per_process_extent, procs_per_dim)]):
                    # strong scaling jobs
                    divided_extent = [extent // num_procs
                                      for extent, num_procs in zip(per_process_extent, procs_per_dim)]
                    # noinspection PyShadowingNames
                    for i, extent in enumerate(divided_extent):
                        divided_extent[i] = max(extent, min_extent[i])
                    job = JobDescription(per_process_extent=divided_extent, procs_per_dim=procs_per_dim,
                                         cuda_aware=cuda_aware)
                    jobs_to_run.append(job)

        print(f"{len(jobs_to_run)} jobs for cuda_aware={cuda_aware}")
        return jobs_to_run


    cuda_aware_vals: List[bool] = [True, False]
    if always_cuda_aware:
        cuda_aware_vals = [True]

    generated_scripts_dir = "generated-scripts"
    slurm_error_dir = f"{generated_scripts_dir}/error"
    slurm_output_dir = f"{generated_scripts_dir}/output"
    for dir_to_make in [generated_scripts_dir, slurm_error_dir, slurm_output_dir]:
        if not os.path.isdir(dir_to_make):
            os.mkdir(dir_to_make)

    preamble = build_slurm_gpu_preamble(machine_config, num_gpus, job_name,
                                        email_address=email_address,
                                        account_name=account_name,
                                        time_limit=time_limit,
                                        mail_type=[MailType.BEGIN, MailType.FAIL],
                                        # %a b/c job array, see https://slurm.schedmd.com/job_array.html#file_names
                                        output_file_name=f"{slurm_output_dir}/{job_name}_%a.out",
                                        error_file_name=f"{slurm_error_dir}/{job_name}_%a.err"
                                        )
    build_scripts = ["#!/bin/bash"] + [build_job(brick_shape, vec_shape, cuda_aware)
                                       for cuda_aware in cuda_aware_vals]
    build_script_filename = os.path.abspath(f"{generated_scripts_dir}/build_{job_name}.sh")
    print(f"Writing build script to {build_script_filename}")
    with open(build_script_filename, 'w') as build_script_file:
        build_script_file.write("\n".join(build_scripts))

    # build job script variables
    per_process_extent_var_name = "per_process_extent"
    procs_per_dim_var_name = "procs_per_dim"
    node_local_procs_per_dim_var_name = "node_local_procs_per_dim"
    cuda_aware_var_name = "cuda_aware"
    job_var_names = [per_process_extent_var_name,
                     procs_per_dim_var_name,
                     node_local_procs_per_dim_var_name,
                     cuda_aware_var_name,
                     ]

    jobs: List[JobDescription] = \
        [job for cuda_aware in cuda_aware_vals for job in run_jobs(cuda_aware)]

    # setup job script variable arrays
    per_process_extents = list(map(lambda job: ','.join(map(str, job.per_process_extent)), jobs))
    procs_per_dims = list(map(lambda job: ','.join(map(str, job.procs_per_dim)), jobs))
    node_local_procs_per_dims = list(map(lambda job: ','.join(map(str,
                                                                  job.get_node_local_procs_per_dim(machine_config))
                                                              ),
                                         jobs))
    cuda_aware_values = list(map(lambda job: "ON" if job.cuda_aware else "OFF", jobs))
    job_var_values = [per_process_extents,
                      procs_per_dims,
                      node_local_procs_per_dims,
                      cuda_aware_values,
                      ]
    assert len(job_var_values) == len(job_var_names)


    def make_bash_array(base_var_name, values):
        array_var_name = base_var_name + "_array"
        values_as_array = "("
        indent = None
        for value in values:
            if indent is not None:
                values_as_array += "\n" + " " * indent
            else:
                indent = len(f"{array_var_name}=(")
            values_as_array += '"' + value + '"'
        values_as_array += "\n" + " " * indent + ")"
        return f"{array_var_name}={values_as_array}"

    for name, values in zip(job_var_names, job_var_values):
        environment_setup += "\n" + make_bash_array(name, values)

    # Now select the actual variables
    for name in job_var_names:
        array_name = name + "_array"
        environment_setup += "\n" + f"{name}=${{{array_name}[${{SLURM_ARRAY_TASK_ID}}]}}"

    # Finally, build the job script
    build_dir = get_build_dir(f"${{{cuda_aware_var_name}}}")
    srun_args = [f"-n {num_gpus}", "--cpu-bind=cores"]
    srun_cmd = f"srun {' '.join(srun_args)}"
    mpich_rank_reorder_dir = os.path.abspath(f"{generated_scripts_dir}/reorder_files")
    if not os.path.exists(mpich_rank_reorder_dir):
        os.mkdir(mpich_rank_reorder_dir)
    mpich_rank_reorder_file = f"{mpich_rank_reorder_dir}/MPICH_RANK_ORDER_{num_gpus}gpus_job${{SLURM_ARRAY_TASK_ID}}"
    job_script = f"""export CUDA_VISIBLE_DEVICES=${{SLURM_LOCALID}}
export MPICH_RANK_REORDER_FILE={mpich_rank_reorder_file}
grid_order -C -c ${{{node_local_procs_per_dim_var_name}}} -g ${{{procs_per_dim_var_name}}} > ${{MPICH_RANK_REORDER_FILE}}
{build_dir}/{executable_name} \\
        -d ${{{per_process_extent_var_name}}} \\
        -p ${{{procs_per_dim_var_name}}} \\
        -I 25 -W 2 -a -G {num_gz} \\
        -o {output_file} \\
        || echo \"{'weak ' if weak_scaling else ''}Failed with {num_gpus} gpus, cuda_aware ${{{cuda_aware_var_name}}}, """ \
                 f"""extent ${{{per_process_extent_var_name}}}, procs per dim ${{{procs_per_dim_var_name}}}""" \
                 f""" and {num_gz} ghost-zones\" >> mpi_failures.txt"""

    exec_script_filename = os.path.abspath(f"{generated_scripts_dir}/{job_name}.sh")
    print(f"Writing exec script to {exec_script_filename}")
    with open(exec_script_filename, 'w') as exec_script_file:
        exec_script_file.write("\n".join(["#!/bin/bash", environment_setup, job_script]))

    slurm_script_filename = os.path.abspath(f"{generated_scripts_dir}/{job_name}.slurm")
    print(f"Writing slurm script to {slurm_script_filename}")
    with open(slurm_script_filename, 'w') as slurm_script_file:
        slurm_script_file.write("\n".join([preamble, "# Run job", f"{srun_cmd} {exec_script_filename}"]))

    build_cmd = f"{build_script_filename}"
    submit_file_name = os.path.abspath(f"{generated_scripts_dir}/{job_name}.submit")
    with open(submit_file_name, 'w') as submit_file:
        submit_file.write("\n".join(["#!/bin/bash",
                                     "# Build",
                                     build_cmd,
                                     f"sbatch --array=0-{len(jobs)-1} {slurm_script_filename}",
                                     ]) + "\n")
    print(f"Writing submit file to {submit_file_name}")

    # make files executable
    for file_name in [submit_file_name, slurm_script_filename, exec_script_filename, build_script_filename]:
        st = os.stat(file_name)
        os.chmod(file_name, st.st_mode | stat.S_IEXEC)
