import argparse
import os
import stat
from typing import List

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
    parser.add_argument("-d", "--per-process-domain-size", type=str,
                        help="Per-process domain extent formatted as I,J,K,L,M,N",
                        default="72,32,32,32,32,2")
    parser.add_argument("--num-gz", type=int, help="Number of ghost-zones to use", default=1)
    parser.add_argument("-o", "--output_file", type=str, help="Output file to write to", default=None)
    parser.add_argument("-J", "--job-name", type=str, help="Name of job", default=None)
    parser.add_argument("-w", "--weak", action='store_true', help="Perform weak scaling instead of strong scaling")
    parser.add_argument("--always-cuda-aware", action='store_true', help="Only use jobs which use CUDA-Aware MPI")

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

    per_process_extent = tuple(map(int, args["per_process_domain_size"].split(',')))
    num_gz = args["num_gz"]

    output_file = args["output_file"]
    if output_file is None:
        output_file = f"mpi_{num_gpus}gpus.csv"
    job_name = args["job_name"]
    if job_name is None:
        job_name = f"mpi_{num_gpus}"

    weak_scaling: bool = args["weak"]
    always_cuda_aware: bool = args["always_cuda_aware"]

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
    environment_setup = f"""
if [[ ! -f "{build_dir}" ]] ; then
    mkdir -p "{build_dir}"
fi
"""


    def get_build_dir(use_types: bool, cuda_aware: bool) -> str:
        return os.path.abspath(f"{build_dir}/use_types_{'ON' if use_types else 'OFF'}_cuda_aware_" +
                               f"{'ON' if cuda_aware else 'OFF'}")


    def build_job(brick_dim, vec_dim, use_types: bool, cuda_aware: bool):
        current_build_dir = get_build_dir(use_types, cuda_aware)
        return f"""
if [[ ! -d {current_build_dir} ]] ; then 
    mkdir -p {current_build_dir}
    echo "Building brick-dim {brick_dim} with vec dim {vec_dim}" ; 
    cmake -S {os.path.abspath(os.path.join(os.path.pardir, os.path.pardir))} \\
        -B {current_build_dir} \\
        -DCMAKE_CUDA_ARCHITECTURES={machine_config.cuda_arch} \\
        -DCMAKE_INSTALL_PREFIX=bin \\
        -DGENE6D_USE_TYPES={"ON" if use_types else "OFF"} \\
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


    environment_setup += f"""#https://docs.nersc.gov/jobs/affinity/#openmp-environment-variables
export OMP_PLACES=threads
export OMP_PROC_BIND=true
export gtensor_DIR={gtensor_dir}
export CUDA_VISIBLE_DEVICES=${{SLURM_LOCALID}}
"""


    class JobDescription:
        """
        A description of a job to run
        """

        # noinspection PyShadowingNames
        def __init__(self, per_process_extent, procs_per_dim, use_types: bool, cuda_aware: bool):
            global num_gz
            global output_file
            self.per_process_extent = tuple(per_process_extent)
            self.procs_per_dim = tuple(procs_per_dim)
            self.num_gz = num_gz
            self.output_file = output_file
            self.build_dir = get_build_dir(use_types, cuda_aware)

        def __str__(self):
            return f"""{self.build_dir}/{executable_name} \\
        -d {','.join(map(str, self.per_process_extent))} \\
        -p {','.join(map(str, self.procs_per_dim))} \\
        -I 25 -W 2 -a -G {self.num_gz} \\
        -o {self.output_file} \\
        || echo \"Failed with {num_gpus} gpus, cuda_aware {cuda_aware}, use_types {use_types}, """ \
                   f"""extent {self.per_process_extent}, procs per dim {self.procs_per_dim}""" \
                   f""" and {self.num_gz} ghost-zones\" >> mpi_failures.txt"""

        def __eq__(self, that):
            return self.per_process_extent == that.per_process_extent \
                   and self.procs_per_dim == that.procs_per_dim \
                   and self.num_gz == that.num_gz

        def __ne__(self, that):
            return not (self == that)

        def __hash__(self):
            return hash((self.per_process_extent, self.procs_per_dim, self.num_gz))


    def run_jobs(use_types: bool, cuda_aware: bool) -> List[JobDescription]:
        jobs_to_run = []
        ghost_depth = (0, 0, 2 * num_gz, 2 * num_gz, 0, 0)
        min_extent = tuple(map(lambda g: max(1, 3 * g), ghost_depth))
        for k in range(1, num_gpus + 1):
            for ell in filter(lambda x: num_gpus % (x * k) == 0, range(1, num_gpus + 1)):
                m = num_gpus // (k * ell)
                assert m * k * ell == num_gpus
                procs_per_dim = [1, 1, k, ell, m, 1]
                if weak_scaling:
                    # weak scaling jobs:
                    job = JobDescription(per_process_extent=per_process_extent, procs_per_dim=procs_per_dim,
                                         use_types=use_types, cuda_aware=cuda_aware)
                    jobs_to_run.append(job)
                elif all([extent % num_procs == 0 for extent, num_procs in zip(per_process_extent, procs_per_dim)]):
                    # strong scaling jobs
                    divided_extent = [extent // num_procs
                                      for extent, num_procs in zip(per_process_extent, procs_per_dim)]
                    # noinspection PyShadowingNames
                    for i, extent in enumerate(divided_extent):
                        divided_extent[i] = max(extent, min_extent[i])
                    job = JobDescription(per_process_extent=divided_extent, procs_per_dim=procs_per_dim,
                                         use_types=use_types, cuda_aware=cuda_aware)
                    if all([any([x < y for x, y in zip(job.per_process_extent, other_job.per_process_extent)])
                            for other_job in jobs_to_run]):
                        jobs_to_run.append(job)

        return jobs_to_run


    use_types_cuda_aware_vals = [(False, True), (False, False), (True, False), (True, True)]
    if always_cuda_aware:
        use_types_cuda_aware_vals = list(filter(lambda x: x[1], use_types_cuda_aware_vals))

    build_scripts = [build_job(brick_shape, vec_shape, use_types, cuda_aware)
                     for use_types, cuda_aware in use_types_cuda_aware_vals]
    job_scripts: List[List[JobDescription]] = \
        [run_jobs(use_types, cuda_aware) for use_types, cuda_aware in use_types_cuda_aware_vals]

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
                                        output_file_name=f"{slurm_output_dir}/{job_name}.out",
                                        error_file_name=f"{slurm_error_dir}/{job_name}.err"
                                        )
    build_scripts = ["#!/bin/bash"] + [build_job(brick_shape, vec_shape, use_types, cuda_aware)
                                       for use_types, cuda_aware in use_types_cuda_aware_vals]
    build_script_filename = os.path.abspath(f"{generated_scripts_dir}/build_{job_name}.sh")
    print(f"Writing build script to {build_script_filename}")
    with open(build_script_filename, 'w') as build_script_file:
        build_script_file.write("\n".join(build_scripts))

    job_script = []
    for (use_types, cuda_aware), jobs in zip(use_types_cuda_aware_vals, job_scripts):
        for job in jobs:
            job_script += list(map(str, run_jobs(use_types, cuda_aware)))
    job_script = ["#!/bin/bash"] +\
                 [f"echo \" Job {i+1} / {len(job_script)}\"\n" + job for i, job in enumerate(job_script)]

    job_script_filename = os.path.abspath(f"{generated_scripts_dir}/{job_name}.sh")
    print(f"Writing executable script to {job_script_filename}")
    with open(job_script_filename, 'w') as job_script_file:
        job_script_file.write("\n".join(job_script))
    # make job script and build script files executable
    for file_name in [job_script_filename, build_script_filename]:
        st = os.stat(file_name)
        os.chmod(file_name, st.st_mode | stat.S_IEXEC)

    build_cmd = f"{build_script_filename}"
    srun_args = [f"-n {num_gpus}", "--cpu-bind=cores"]
    srun_cmd = f"srun {' '.join(srun_args)} {job_script_filename}"
    slurm_script_filename = os.path.abspath(f"{generated_scripts_dir}/{job_name}.slurm")
    print(f"Writing slurm script to {slurm_script_filename}")
    with open(slurm_script_filename, 'w') as slurm_script_file:
        slurm_script_file.write("\n".join([preamble, environment_setup, build_cmd, srun_cmd]))
