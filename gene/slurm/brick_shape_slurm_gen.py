import argparse
import os
import random

from math import gcd

from slurmpy import *

if __name__ == "__main__":
    known_machines = [machine for machine in machine_configurations]
    parser = argparse.ArgumentParser("Build slurm job file for Single-device Gene brick-shape experiment")
    parser.add_argument("bricklib-src-dir", type=str, help="Path to bricklib source directory")
    parser.add_argument("-J", "--job_name", type=str, help="Job name", default=None)
    parser.add_argument('-M', "--machine", type=str, help=f"Machine name, one of {known_machines}")
    parser.add_argument("-e", "--email", type=str, help="Email address to notify on job completion")
    parser.add_argument("-i", "--image", type=str, help="Shifter image to use", default=None)
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
    image_name = args["image"]
    bricklib_src_dir = os.path.abspath(args["bricklib_src_dir"])

    preamble = build_slurm_gpu_preamble(config=machine_config,
                                        num_gpus=1,
                                        job_name=job_name,
                                        email_address=email_address,
                                        account_name=account_name,
                                        time_limit=time_limit,
                                        image_name=image_name,
                                        mail_type=[MailType.BEGIN, MailType.FAIL],
                                        )

    per_process_extent = tuple(map(int, args["per_process_domain_size"].split(',')))

    def get_range(dim, remaining_size):
        if dim in [0, 2, 3]:
            is_valid_extent = lambda x: (per_process_extent[dim] + 2 * (x-2)) % x == 0 and remaining_size % x == 0
            min_val = 2
        else:
            is_valid_extent = lambda x: per_process_extent[dim] % x == 0 and remaining_size % x == 0
            min_val = 1
        max_val = min(per_process_extent[dim], remaining_size)
        return filter(is_valid_extent, range(min_val, max_val + 1))

    brick_dims = []
    vector_dims = []
    brick_sizes = [64, 128, 256, 512, 1024]
    for i in get_range(0, max(brick_sizes)):
        for k in get_range(2, max(brick_sizes) // i):
            for ell in get_range(3, max(brick_sizes) // i // k):
                for brick_size in filter(lambda x: x % (i*k*ell) == 0, brick_sizes):
                    jmn_size = brick_size // i // k // ell
                    for j in get_range(1, jmn_size):
                        mn_size = jmn_size // j
                        n = gcd(mn_size, per_process_extent[5])
                        m = mn_size // n
                        if per_process_extent[4] % m == 0:
                            assert i * j * k * ell * m * n == brick_size
                            brick_dim = (i, j, k, ell, m, n)
                            vec_dim = [1, 1, 1, 1, 1, 1]
                            vec_len = 1
                            d = 0
                            while vec_len < 32 and d < 6:
                                vec_dim[d] = gcd((32 // vec_len), brick_dim[d])
                                vec_len *= vec_dim[d]
                                d += 1

                            if vec_len == 32:
                                brick_dims.append((i, j, k, ell, m, n))
                                vector_dims.append(tuple(vec_dim))

    build_dir = f"cmake-builds/single/{machine_config.name}"
    preamble += f"""
if [[ ! -f "{build_dir}" ]] ; then
    mkdir -p "{build_dir}"
fi
"""
    brick_dim_var_name = "brick_dim"
    shifter_args = "" if image_name is None else "shifter --module=gpu"

    def build_job(brick_dim, vec_dim):
        return f"""echo "Building brick-dim {brick_dim} with vec dim {vec_dim}" ; 
    cmake -S {bricklib_src_dir} \\
        -B {build_dir} \\
        -DCMAKE_CUDA_ARCHITECTURES={machine_config.cuda_arch} \\
        -DCMAKE_INSTALL_PREFIX=bin \\
        -DGENE6D_USE_TYPES=OFF \\
        -DGENE6D_CUDA_AWARE=OFF \\
        -DGENE6D_BRICK_DIM={','.join(map(str,reversed(brick_dim)))} \\
        -DGENE6D_VEC_DIM={','.join(map(str,reversed(vec_dim)))} \\
        -DCMAKE_CUDA_FLAGS=\"--resource-usage -lineinfo -gencode arch=compute_{machine_config.cuda_arch},code=[sm_{machine_config.cuda_arch},lto_{machine_config.cuda_arch}]\" \\
        -DCMAKE_BUILD_TYPE=Release \\
        -DPERLMUTTER={"ON" if machine_config.name == "perlmutter" else "OFF"} \\
    || exit 1
    (cd {build_dir} && make clean && make -j 20 single-gene-6d 2> "${{ptx_info_file}}") || exit 1
    python3 get_ptx_info.py
"""

    def run_job(extent, first_job: bool):
        tmp_file_name = "tmp" + str(random.randint(0, 1000000))
        return f"""echo "Running experiment with extent {extent}"
    srun -n 1 {shifter_args} {build_dir}/gene/single-gene6d -d {','.join(map(str,extent))} \\
        -o {args["output_file"]} \\
        -a \\
        -I 100 \\
        -W 5 \\
    || exit 1
    
    # Create file that runs command, piping output to dev/null
    echo "#!/bin/bash
    {build_dir}/gene/single-gene6d -d {','.join(map(str,extent))} \\
    -o {tmp_file_name}.tmp \\
    -I 1 \\
    -W 0 \\
    > /dev/null
rm {tmp_file_name}.tmp" > {tmp_file_name}.run
    chmod 777 {tmp_file_name}.run
    
    # Use backticks around comments inside multiline command
    # https://stackoverflow.com/questions/9522631/how-to-put-a-line-comment-for-a-multi-line-command
    
    # Now profile that file using ncu, collecting metrics into csv
    srun -n 1 {shifter_args} ncu --csv   `# output data as csv to console` \\
        --target-processes all `# profile child processes` \\
        --metrics $metncu11,$register_metrics,$l1_load_metrics,$lts_load_metrics,$occupancy_metrics \\
        {tmp_file_name}.run \\
        | sed '/^==PROF==/d'   `# Remove lines above csv output by filter on regex "^==PROF=="` \\
        > {tmp_file_name}.csv \\
    || exit 1
    rm {tmp_file_name}.run
    {"mv" if first_job else "tail -n +2 "} {tmp_file_name}.csv {"" if first_job else ">>"} {"ncu_" + args["output_file"]} 
    {"" if first_job else f"rm {tmp_file_name}.csv"}
"""

    all_jobs = []
    for brick_dim, vec_dim in zip(brick_dims, vector_dims):
        first_job = len(all_jobs) == 0
        all_jobs.append(build_job(brick_dim, vec_dim))
        extent = list(per_process_extent)
        all_jobs.append(run_job(extent, first_job))

    # from
    # https://github.com/cyanguwa/nersc-roofline/blob/65487eb44c1290f195cf2edb67cc329614aa86ae/GPP/Volta/run.survey
    l1_load_metrics = ["l1tex__t_bytes"] + [f"l1tex__t_sectors_pipe_lsu_mem_{memory_space}_op_{access_type}"
                                            for memory_space in ["global", "local"] for access_type in ["ld", "st"]
                                            ]
    lts_load_metrics = ["lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_local_op_ld",
                        "lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_local_op_st",
                        "lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_st",
                        "lts__t_bytes_equiv_l1sectormiss_pipe_lsu_mem_global_op_ld"
                        ]
    occupancy_metrics = ["sm__warps_active.avg.pct_of_peak_sustained_active"]
    environment_setup = [
        "ptx_info_file=$(pwd)/ptx_info.txt",
        "metncu11='sm__cycles_elapsed.avg,sm__cycles_elapsed.avg.per_second,"
        "sm__sass_thread_inst_executed_op_dadd_pred_on.sum,sm__sass_thread_inst_executed_op_dfma_pred_on.sum,"
        "sm__sass_thread_inst_executed_op_dmul_pred_on.sum,sm__sass_thread_inst_executed_op_fadd_pred_on.sum,"
        "sm__sass_thread_inst_executed_op_ffma_pred_on.sum,sm__sass_thread_inst_executed_op_fmul_pred_on.sum,"
        "sm__sass_thread_inst_executed_op_hadd_pred_on.sum,sm__sass_thread_inst_executed_op_hfma_pred_on.sum,"
        "sm__sass_thread_inst_executed_op_hmul_pred_on.sum,sm__inst_executed_pipe_tensor.sum,l1tex__t_bytes.sum,"
        "lts__t_bytes.sum,dram__bytes.sum'",
        "register_metrics='launch__registers_per_thread'",
        "l1_load_metrics='" + ".sum,".join(l1_load_metrics) + ".sum'",
        "lts_load_metrics='" + ".sum,".join(lts_load_metrics) + ".sum'",
        "occupancy_metrics='" + ','.join(occupancy_metrics) + "'",
    ]
    slurm_script = "\n\n".join([preamble] + environment_setup + all_jobs) + "\n"

    print(slurm_script)
    print(f"# {len(brick_dims)} jobs total")

