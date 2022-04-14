# GENE Benchmarks

This directory contains benchmarks based on the [GENE code](http://genecode.org)
and the [GTensor](https://github.com/wdmapp/gtensor) project used
to port GENE to GPU 
(see [Toward exascale whole-device modeling of fusion devices: Porting the GENE gyrokinetic microturbulence code to GPU](https://aip.scitation.org/doi/10.1063/5.0046327)
 in Physics of Plasmas, 2021).

## Benchmark Structure

This directory builds three benchmarks:
* `single-gene-6d`: Runs the stencil experiments on a single GPU for both GENE and Bricks implementations
* `mpi-gene6d`: Runs the 2D stencil experiment across multiple GPUs for both GENE and Bricks implementations
* `fft-gene-6d`: Computes a 1-dimensional FFT along the j-axis for both array and Bricks layouts using cuFFT

## Setup

For our SC 2022 submission, we have automated the setup using [Docker](https://www.docker.com)
and [Shifter](https://docs.nersc.gov/development/shifter/) through NERSC.
The Dockerfile is located in the `docker` directory, if you wish to see the setup for yourself.
To replicate the results, please follow [the Shifter setup instructions](#setup-using-shifter-on-perlmutter)
below. Then, follow the instructions in the [running the benchmarks section](#running-the-benchmarks) below.

### Setup using Shifter on Perlmutter

These instructions are used to set up the results on the [NERSC Perlmutter cluster](https://docs.nersc.gov/systems/perlmutter/).
If you do not have access to perlmutter,
you can apply as described on [NERSC's website](https://docs.nersc.gov/systems/perlmutter/#access).

Begin by logging into perlmutter.
```bash
ssh perlmutter
```
Next, use `shifter` to set up the environment.
First, pull the image down from DockerHub. This may a few minutes.
```bash
shifterimg -v pull bensepanski/2022_sc_bricks:0.1
```
Next, start the container and move to the `bricklib` directory.
```bash
shifter --image=bensepanski/2022_sc_bricks:0.1 --module=gpu --module=cuda-mpich /bin/bash
cd /bricks/sc_22_submission/bricklib/
```
You should see all the directories and files in the `bricklib` library,
including a pre-built `build/` directory and the source directory
for these benchmarks (i.e. `gene/`).

### Manual setup

FIXME: DESCRIBE MANUAL SETUP HERE

## Running the benchmarks

We assume you are starting in this directory, i.e. in `bricklib/gene`.
Our benchmarks are set up to run using [slurm](https://docs.nersc.gov/jobs/).
Benchmarks can be run [interactively](#running-jobs-interactively) to play with the benchmarks
yourself, or using [batch jobs](#batch-jobs) to run the same experiments as in the paper.

### Running jobs interactively

First, (outside the shifter image, if you are using the containerized setup)
get a compute node interactively using salloc as described in the [NERSC docs](https://docs.nersc.gov/jobs/).
For example, on perlmutter you might run
```bash
salloc -C gpu -N 1 -G 4 -n 4 -c 20 -t 00:30:00 -q interactive
```
to obtain a single node for 30 minutes. If you are using the [shifter image](#setup-using-shifter-on-perlmutter),
activate it and move to the build directory.
```bash
shifter --image=bensepanski/2022_sc_bricks:0.1 --module=gpu --module=cuda-mpich /bin/bash
cd /bricks/sc_22_submission/bricklib/build/
```

FIXME: DESCRIBE EXTRA SHIFTER SETUP FOR CUDA-AWARE MPI

FIXME: DESCRIBE HOW TO RUN ALL 3 BENCHMARKS


### Batch jobs

FIXME: DESCRIBE HOW TO GENERATE JOBS

FIXME: DESCRIBE HOW TO SUBMIT JOBS

FIXME: DESCRIBE HOW TO GET RESULTS