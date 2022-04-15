import argparse
import os

from slurmpy import *

if __name__ == "__main__":
    known_machines = [machine for machine in machine_configurations]
    parser = argparse.ArgumentParser("Build slurm job file for Single-device Gene FFT experiment")
    parser.add_argument("bricklib-src-dir", type=str, help="Path to bricklib source directory")
    parser.add_argument("-J", "--job_name", type=str, help="Job name", default=None)
    parser.add_argument('-M', "--machine", type=str, help=f"Machine name, one of {known_machines}")
    parser.add_argument("-e", "--email", type=str, help="Email address to notify on job completion")
    parser.add_argument("-i", "--image", type=str, help="Shifter image to use", default=None)
    parser.add_argument("-A", "--account", type=str, help="slurm account to charge")
    parser.add_argument("-t", "--time-limit", type=str, help="Time limit, in format for slurm", default="01:00:00")
    parser.add_argument("-o", "--output_file", type=str, help="Output file to write to", default="fft.csv")

    args = vars(parser.parse_args())

    machine_name = args["machine"]
    if machine_name not in machine_configurations:
        raise ValueError(f"Unrecognized machine {machine_name}, must be one of {known_machines}")
    machine_config = machine_configurations[machine_name]
    job_name = args["job_name"]
    if job_name is None:
        job_name = "single-gene-" + machine_name + f"-fft"

    time_limit = args["time_limit"]
    account_name = args["account"]
    email_address = args["email"]
    image_name = args["image"]
    bricklib_src_dir = os.path.abspath(args["bricklib-src-dir"])

    preamble = build_slurm_gpu_preamble(config=machine_config,
                                        num_gpus=1,
                                        job_name=job_name,
                                        email_address=email_address,
                                        account_name=account_name,
                                        time_limit=time_limit,
                                        image_name=image_name,
                                        mail_type=[MailType.BEGIN, MailType.FAIL],
                                        )

    brick_dims = [(2, 32, 2, 2, 1, 1),
                  (2, 16, 2, 2, 1, 1),
                  (2, 8, 4, 2, 1, 1),
                  (2, 4, 4, 4, 1, 1),
                  (2, 2, 8, 4, 1, 1)]
    vec_dims = [(2, 16, 1, 1, 1, 1),
                (2, 16, 1, 1, 1, 1),
                (2, 8, 2, 1, 1, 1),
                (2, 4, 4, 1, 1, 1),
                (2, 2, 8, 1, 1, 1)
                ]

    build_dir = os.path.abspath(f"cmake-builds/single/{machine_config.name}")
    preamble += f"""
if [[ ! -f "{build_dir}" ]] ; then
    mkdir -p "{build_dir}"
fi
"""
    shifter_args = "" if image_name is None else "shifter --module=gpu"

    def build_job(brick_dim, vec_dim):
        return f"""{shifter_args} cmake -S {bricklib_src_dir} \\
        -B {build_dir} \\
        -DCMAKE_CUDA_ARCHITECTURES={machine_config.cuda_arch} \\
        -DCMAKE_INSTALL_PREFIX=bin \\
        -DGENE6D_USE_TYPES=OFF \\
        -DGENE6D_CUDA_AWARE=OFF \\
        -DGENE6D_BRICK_DIM={','.join(map(str,reversed(brick_dim)))} \\
        -DGENE6D_VEC_DIM={','.join(map(str,reversed(vec_dim)))} \\
        -DCMAKE_CUDA_FLAGS=\"-lineinfo -gencode arch=compute_{machine_config.cuda_arch},code=[sm_{machine_config.cuda_arch},lto_{machine_config.cuda_arch}]\" \\
        -DCMAKE_BUILD_TYPE=Release \\
        -DPERLMUTTER={"ON" if machine_config.name == "perlmutter" else "OFF"} \\
    || exit 1
    {shifter_args} "(cd {build_dir} && make clean && make -j 20 single-fft-gene-6d) || exit 1"
"""

    def run_job(brick_dim = None):
        if brick_dim is None:
            to_run = "a"
        else:
            to_run = "b"
        return f"""echo "Running experiment with brick size {brick_dim}"
    srun -n 1 {shifter_args} {build_dir}/gene/fft-gene-6d 10 100 {to_run} || exit 1"""

    all_jobs = []
    for brick_dim, vec_dim in zip(brick_dims, vec_dims):
        all_jobs.append(build_job(brick_dim, vec_dim))
        all_jobs.append(run_job(brick_dim))

    slurm_script = "\n\n".join([preamble] + all_jobs) + "\n"

    print(slurm_script)

