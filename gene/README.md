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

To use the same modules and environment setup outside of the container, run the [`perlmutter_setup.sh`](https://github.com/benSepanski/bricklib/blob/sc-22-artifact/gene/docker/perlmutter_setup.sh) script.
```bash
source perlmutter_setup.sh
```

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
export BRICKS_SHIFTER_IMG=docker:bensepanski/2022_sc_bricks:perlmutter_0.1
shifterimg -v pull ${BRICKS_SHIFTER_IMG}
```
Next, start the container and move to the `bricklib` directory.
```bash
export BRICKS_SHIFTER_ARGS="--entrypoint --module=gpu --module=cuda-mpich"
shifter --image=${BRICKS_SHIFTER_IMG} ${BRICKS_SHIFTER_ARGS} /bin/bash
```
You should now be in the [`${SCRATCH}`](https://docs.nersc.gov/filesystems/#scratch) directory
which contains a single directory: `bricklib`, containing pre-built executables in `bricklib/build`.
Use the
```bash
exit
```
command to exit the container.

*Note:*
Since image directories are mounted read-only at NERSC ([docs](https://docs.nersc.gov/development/shifter/how-to-use/#differences-between-shifter-and-docker),
you won't be able to make any changes in the directory.
Instead, we'll have to run everything from outside the `bricklib` directory.
Feel free to move to a different directory if you wish to store the run output in a more persistent
directory than scratch.

Now you're ready to start [running the benchmarks](#running-the-benchmarks)!

### Manual setup

Make sure you've run the setup file [`perlmutter_setup.sh`](https://github.com/benSepanski/bricklib/blob/sc-22-artifact/gene/docker/perlmutter_setup.sh)
script to ensure you have the necessary modules.

First, download the GTensor library and checkout the commit used in our experiments.
```bash
WORK_DIR="`pwd`"
git clone https://github.com/wdmapp/gtensor.git
cd gtensor 
git reset --hard 41cf4fe26625f8d7ba2d0d3886a54ae6415a2017 
```
Then, install the GTensor library using cmake.
```bash
cmake -S . -B build -DGTENSOR_DEVICE=cuda \
  -DCMAKE_INSTALL_PREFIX=bin \
  -DBUILD_TESTING=OFF \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build --target install
```
We also need to set an environment variable to help CMake find GTensor.
```bash
gtensor_DIR="${WORK_DIR}/gtensor/bin"
````

Next, download the Bricks library.
```bash
cd ${WORK_DIR}
git clone https://github.com/benSepanski/bricklib.git # (While waiting for PR to work its way through bitbucket, pull from fork)
cd bricklib
git checkout 5abff1121025a04858f4b85ec2260de435169b27 # TODO: Switch to tag once finalized
```
For consistency with the instructions for the shifter image instructions, go ahead and
store the location of the Bricks library in the environment variable `bricklib_SRCDIR`.
```bash
bricklib_SRCDIR="`pwd`"
```
We'll assume you are building on the Perlmutter machine.
```bash
export BUILDING_IMAGE_ON_PERLMUTTER=ON
export CUDA_ARCHITECTURE=80
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=bin \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURE}" \
  -DGENE6D_USE_TYPES=OFF  \
  -DGENE6D_CUDA_AWARE=ON \
  -DPERLMUTTER=${BUILDING_IMAGE_ON_PERLMUTTER} \
  -DCMAKE_CUDA_FLAGS="-lineinfo -gencode arch=compute_${CUDA_ARCHITECTURE},code=[sm_${CUDA_ARCHITECTURE},lto_${CUDA_ARCHITECTURE}]"
cmake --build build --parallel
```

Now you're ready to start [running the benchmarks](#running-the-benchmarks)!

## Running the benchmarks

We assume you are starting in this directory, i.e. in `bricklib/gene`.
Our benchmarks are set up to run using [slurm](https://docs.nersc.gov/jobs/).
Benchmarks can be run [interactively](#running-jobs-interactively) to play with the benchmarks
yourself, or using [batch jobs](#batch-jobs) to run the same experiments as in the paper.

### Running jobs interactively

First, (outside the shifter image, if you are using the containerized setup)
get a compute node interactively using salloc as described in the [NERSC docs](https://docs.nersc.gov/jobs/).
If you are using the  [shifter image](#setup-using-shifter-on-perlmutter) setup,
you must use the extra argument `--image=bensepanski/2022_sc_bricks:perlmutter_0.1`,
and run the `shifter` image.
For example, on perlmutter you might run
```bash
# Shifter build:
salloc -C gpu -N 1 -G 4 -n 4 -c 20 -t 00:30:00 -q interactive --image=${BRICKS_SHIFTER_IMG}
shifter ${BRICKS_SHIFTER_ARGS} /bin/bash
# Manual build:
salloc -C gpu -N 1 -G 4 -n 4 -c 20 -t 00:30:00 -q interactive
```

Now we can run the benchmarks interactively.
We'll assume the Bricks library source is stored in `bricklib_SRCDIR` and has been
built in `$bricklib_SRCDIR/build`.
If you used the shifter image, all three benchmarks in the `build/` directory are already on your `PATH`.
Otherwise, you need to move to the `${bricklib_SRCDIR}/build/gene` directory.
```bash
# Shifter build:
# Nothing to do!
# Manual build:
cd "${bricklib_SRCDIR}/build/gene"
```

#### Single-Device Stencils

To run the `single-gene-6d` benchmark you only need to specify the array extents.
For example, you can run the benchmark with an `I x J x K x L x M x N` grid
of shape `72 x 32 x 24 x 24 x 32 x 2` (including ghost zones) by running
```bash
single-gene-6d -d 72,32,24,24,32,2
```
To see further command line options, run
```bash
single-gene-6d -h
```

#### FFT

To run the FFT benchmark, run
```bash
fft-gene-6d
```
If you wish, you can specify the number of warmups iterations/measured iterations.
For instance, to run with 5 warmup iterations and 100 measured iterations, run
```bash
fft-gene-6d 5 100
```
You can also choose to run only the array implementation, or only the bricks implementation.
```bash
fft-gene-6d 5 100 a  # Run array layout only
fft-gene-6d 5 100 b  # Run bricks layout only
```

#### MPI Scaling for 2D Stencil

To run the `mpi-gene6d` benchmark you need to specify the array extents
and the number of MPI ranks along each axis.
Using the shifter setup, you'll need to run the benchmark from outside
of the container to use more than one MPI rank.
For example, you can run the benchmark with a global `I x J x K x L x M x N` grid
of shape `72 x 32 x 24 x 24 x 32 x 2` (including ghost zones) split across
2 MPI ranks in the k axis and 2 MPI ranks in the l axis by running
```bash
# Shifter build: (NB: Don't use ${BRICKS_SHIFTER_ARGS} here! It will run from ENTRYPOINT instead of mpi-gene6d)
srun -n 4 shifter --module=gpu --module=cuda-mpich mpi-gene6d -d 72,32,24,24,32,2 -p 1,1,2,2,1,1
# Manual build:
srun -n 4 mpi-gene6d -d 72,32,8,8,32,2 -p 1,1,2,2,1,1
```
To see further command line options, run
```bash
mpi-gene6d -h
```

### Batch jobs

***FIXME: DESCRIBE HOW TO GENERATE JOBS***

***FIXME: DESCRIBE HOW TO SUBMIT JOBS***

***FIXME: DESCRIBE HOW TO GET RESULTS***