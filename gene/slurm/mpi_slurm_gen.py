import argparse
import os

from slurmpy import *


if __name__ == "__main__":
    known_machines = [machine for machine in machine_configurations]
    parser = argparse.ArgumentParser("Build slurm job file for MPI Gene")
    parser.add_argument("gpus", metavar="G", type=int, help="Number of GPUs")
    parser.add_argument("brick_shape", metavar="BI,BJ,BK,BL,BM,BN", type=str, help="Brick shape")
    parser.add_argument("-J", "--job_name", type=str, help="Job name", default=None)
    parser.add_argument('-M', "--machine", type=str, help=f"Machine name, one of {known_machines}")
    parser.add_argument("-e", "--email", type=str, help="Email address to notify on job completion")
    parser.add_argument("-A", "--account", type=str, help="slurm account to charge")
    parser.add_argument("-t", "--time-limit", type=str, help="Time limit, in format for slurm", default="01:00:00")
    parser.add_argument("--compiler", type=str, help="Compiler being used for affinity check", default="gnu")
    parser.add_argument("-d", "--per-process-domain-size", type=str, 
                        help="Per-process domain extent formatted as I,J,K,L,M,N",
                        default="72,32,24,24,32,2")
    parser.add_argument("--num-gz", type=int, help="Number of ghost-zones to use", default=1)
    parser.add_argument("-o", "--output_file", type=str, help="Output file to write to", default=None)

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
    brick_shape = tuple(map(int, args["brick_shape"].split(",")))
    assert len(brick_shape) == 6
    assert all([b > 0 for b in brick_shape])

    vec_len = 1
    vec_shape = [None for _ in range(len(brick_shape))]
    for i in range(len(brick_shape)):
        vec_shape[i] = max(min(brick_shape[i], 32 // vec_len), 1)
        vec_len *= vec_shape[i]
    assert vec_len == 32

    preamble = build_slurm_gpu_preamble(machine_config, num_gpus, job_name,
                                        email_address=email_address,
                                        account_name=account_name,
                                        time_limit=time_limit)

    gtensor_dir = os.getenv("gtensor_DIR")
    if gtensor_dir is None:
        raise ValueError("Environment variable gtensor_DIR must be set")

    build_dir = f"cmake-builds/mpi/{machine_config.name}"
    preamble += f"""
if [[ ! -f "{build_dir}" ]] ; then
    mkdir -p "{build_dir}"
fi
"""

    def build_job(brick_dim, vec_dim, use_types: bool, cuda_aware: bool):
        return f"""echo "Building brick-dim {brick_dim} with vec dim {vec_dim}" ; 
    cmake -S ../../ \\
        -B {build_dir} \\
        -DCMAKE_CUDA_ARCHITECTURES={machine_config.cuda_arch} \\
        -DCMAKE_INSTALL_PREFIX=bin \\
        -DGENE6D_USE_TYPES={"ON" if use_types else "OFF"} \\
        -DGENE6D_CUDA_AWARE={"ON" if cuda_aware else "OFF"} \\
        -DGENE6D_BRICK_DIM={','.join(map(str,reversed(brick_dim)))} \\
        -DGENE6D_VEC_DIM={','.join(map(str,reversed(vec_dim)))} \\
        -DCMAKE_CUDA_FLAGS=\"--resource-usage -lineinfo -gencode arch=compute_{machine_config.cuda_arch},code=[sm_{machine_config.cuda_arch},lto_{machine_config.cuda_arch}]\" \\
        -DCMAKE_BUILD_TYPE=Release \\
        -DPERLMUTTER={"ON" if machine_config.name == "perlmutter" else "OFF"} \\
    || exit 1
    (cd {build_dir} && make clean && make -j 20 mpi-gene6d || exit 1
"""

    preamble += f"""#https://docs.nersc.gov/jobs/affinity/#openmp-environment-variables
export OMP_PLACES=threads
export OMP_PROC_BIND=true
export gtensor_DIR={gtensor_dir}
"""

    num_gz = args["num_gz"]
    output_file = args["output_file"]
    if output_file is None:
        output_file = job_name + ".csv"


    class JobDescription:
        """
        A description of a job to srun
        """
        def __init__(self, per_process_extent, procs_per_dim):
            global num_gz
            global output_file
            self.per_process_extent = tuple(per_process_extent)
            self.procs_per_dim = tuple(procs_per_dim)
            self.num_gz = num_gz
            self.output_file = output_file

        def __str__(self):
            srun_cmd = "srun --cpu-bind=cores"
            return f"{srun_cmd} ${{bricks_DIR}}/gene/mpi-gene6d \
        -d {','.join(map(str,self.per_process_extent))} \
        -p {','.join(map(str,self.procs_per_dim))} \
        -I 100 -W 5 -a -G {self.num_gz} \
        -o {self.output_file}"

        def __eq__(self, that):
            return self.per_process_extent == that.per_process_extent \
                and self.procs_per_dim == that.procs_per_dim \
                and self.num_gz == that.num_gz

        def __ne__(self, that):
            return not(self == that)

        def __hash__(self):
            return hash((self.per_process_extent, self.procs_per_dim, self.num_gz))

    per_process_extent = tuple(map(int, args["per_process_domain_size"].split(',')))
    
    # Generate the jobs
    strong_scaling_jobs = []
    weak_scaling_jobs = []
    min_extent = [1, 1, 3 * (2 * num_gz), 3 * (2 * num_gz), 1, 1]
    for k in range(1, num_gpus+1):
        for ell in filter(lambda x: num_gpus % (x * k) == 0, range(1, num_gpus+1)):
            m = num_gpus // (k * ell)
            assert m * k * ell == num_gpus
            # weak scaling jobs:
            procs_per_dim = [1, 1, k, ell, m, 1]
            job = JobDescription(per_process_extent=per_process_extent, procs_per_dim=procs_per_dim)
            weak_scaling_jobs.append(job)
            # strong scaling jobs
            divided_extent = [extent // num_procs for extent, num_procs in zip(per_process_extent, procs_per_dim)]
            for i, extent in enumerate(divided_extent):
                divided_extent[i] = max(extent, min_extent[i])
            job = JobDescription(per_process_extent=divided_extent, procs_per_dim=procs_per_dim)
            if all([any([x < y for x, y in zip(job.per_process_extent, other_job.per_process_extent)]) for other_job in strong_scaling_jobs]):
                strong_scaling_jobs.append(job)

    # Record the jobs
    run_jobs = '\n'.join(
        ["\n# strong scaling jobs"] + list(map(str, strong_scaling_jobs))
        + ["\n\n# weak scaling jobs"] + list(map(str, weak_scaling_jobs))
        )

    slurm_scripts = [preamble]
    for (use_types, cuda_aware) in [(True, False), (False, True), (False, False)]:
        slurm_scripts += [build_job(brick_shape, vec_shape, use_types, cuda_aware), run_jobs]

    slurm_script = "\n\n".join(slurm_scripts) + "\n"

    print(slurm_script)
