import argparse
import os
from slurmpy import *

if __name__ == "__main__":
    known_machines = [machine for machine in machine_configurations]
    parser = argparse.ArgumentParser("Build slurm job file for Single-device Gene brick-shape experiment")
    parser.add_argument("-J", "--job_name", type=str, help="Job name", default=None)
    parser.add_argument('-M', "--machine", type=str, help=f"Machine name, one of {known_machines}")
    parser.add_argument("-e", "--email", type=str, help="Email address to notify on job completion")
    parser.add_argument("-A", "--account", type=str, help="slurm account to charge")
    parser.add_argument("-t", "--time-limit", type=str, help="Time limit, in format for slurm", default="01:00:00")
    parser.add_argument("-d", "--per-process-domain-size", type=str,
                        help="Per-process domain extent formatted as I,J,K,L,M,N",
                        default="72,32,24,24,32,2")
    parser.add_argument("-o", "--output_file", type=str, help="Output file to write to", default="brick_shape_out.csv")

    args = vars(parser.parse_args())

    machine_name = args["machine"]
    if machine_name not in machine_configurations:
        raise ValueError(f"Unrecognized machine {machine_name}, must be one of {known_machines}")
    machine_config = machine_configurations[machine_name]
    job_name = args["job_name"]
    if job_name is None:
        job_name = "single-gene-" + machine_name + f"-brick-shape"

    time_limit = args["time_limit"]
    account_name = args["account"]
    email_address = args["email"]

    preamble = build_slurm_gpu_preamble(config=machine_config,
                                        num_gpus=1,
                                        job_name=job_name,
                                        email_address=email_address,
                                        account_name=account_name,
                                        time_limit=time_limit)

    per_process_extent = tuple(map(int, args["per_process_domain_size"].split(',')))

    brick_dims = [(2, 32, 2, 2, 1, 1), (2, 16, 2, 2, 2, 1), (2, 16, 2, 4, 1, 1), (2, 16, 4, 2, 1, 1),
                  (4, 16, 2, 2, 1, 1), (2, 8, 2, 2, 4, 1), (4, 8, 2, 2, 2, 1), (2, 8, 4, 4, 1, 1), (2, 16, 2, 2, 1, 1),
                  (2, 8, 2, 2, 2, 1), (2, 8, 2, 4, 1, 1), (2, 8, 4, 2, 1, 1), (4, 8, 2, 2, 1, 1), (2, 4, 2, 2, 4, 1),
                  (4, 4, 2, 2, 2, 1), (2, 4, 4, 4, 1, 1)]

    build_dir = f"cmake-builds/single/{machine_config.name}"
    preamble += f"""
if [[ ! -f "{build_dir}" ]] ; then
    mkdir -p "{build_dir}"
fi
"""
    brick_dim_var_name = "brick_dim"
    build_job = f"""cmake -S ../../ \\
        -B {build_dir} \\
        -DCMAKE_CUDA_ARCHITECTURES={machine_config.cuda_arch} \\
        -DCMAKE_INSTALL_PREFIX=bin \\
        -DGENE6D_USE_TYPES=OFF \\
        -DGENE6D_CUDA_AWARE=OFF \\
        -DGENE6D_BRICK_DIM=${{{brick_dim_var_name}}} \\
        -DCMAKE_CUDA_FLAGS=\"-lineinfo -gencode arch=compute_{machine_config.cuda_arch},code=[sm_{machine_config.cuda_arch},lto_{machine_config.cuda_arch}]\" \\
        -DCMAKE_BUILD_TYPE=Release \\
        -DPERLMUTTER={"ON" if machine_config.name == "perlmutter" else "OFF"} \\
    || exit 1
    (cd {build_dir} && make -j 20 single-gene-6d) || exit 1
"""
    run_job = f"""echo "Running experiment with brick size ${{{brick_dim_var_name}}}"
    srun -n 1 {build_dir}/gene/single-gene6d -d {','.join(map(str,per_process_extent))} \\
        -o {args["output_file"]} \\
        -a \\
        -I 100 \\
        -W 10 \\
    || exit 1
"""
    run_all_jobs = f"""for {brick_dim_var_name} in {' '.join(map(lambda dims: ','.join(map(str, dims)), brick_dims))} ; do
    {build_job}
    {run_job}
done
"""

    slurm_script = "\n\n".join([preamble, run_all_jobs]) + "\n"

    print(slurm_script)

