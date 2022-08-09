# GENE Benchmarks

This directory contains benchmarks based on the [GENE code](http://genecode.org)
and the [GTensor](https://github.com/wdmapp/gtensor) project used
to port GENE to GPU 
(see [Toward exascale whole-device modeling of fusion devices: Porting the GENE gyrokinetic microturbulence code to GPU](https://aip.scitation.org/doi/10.1063/5.0046327)
 in Physics of Plasmas, 2021).

If you just want the data and to make the plots yourself, you can download it
at one of the author's [website](https://www.cs.utexas.edu/~bmsepan/projects/highDimensionalBricks/).

## Benchmark Structure

This directory builds three benchmarks:
* `single-gene-6d`: Runs the stencil experiments on a single GPU for both GENE and Bricks implementations
* `fft-gene-6d`: Computes a 1-dimensional FFT along the j-axis for both array and Bricks layouts using cuFFT

## Setup

For our SC 2022 MCHPC submission, we have automated the setup using [Docker](https://www.docker.com)
and [Shifter](https://docs.nersc.gov/development/shifter/) through NERSC.
The Dockerfile is located in the `docker` directory, if you wish to see the setup for yourself.
To replicate the results, please follow [the Shifter setup instructions](#setup-using-shifter-on-perlmutter)
below. Then, follow the instructions in the [running the benchmarks section](#running-the-benchmarks) below.

To use the same modules and environment setup outside the container, run the [`perlmutter_setup.sh`](https://github.com/benSepanski/bricklib/blob/sc-22-artifact/gene/docker/perlmutter_setup.sh) script.
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
Since image directories are mounted read-only at NERSC [docs](https://docs.nersc.gov/development/shifter/how-to-use/#differences-between-shifter-and-docker),
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
git checkout tags/sc22_artifact1.1
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
get a compute node interactively using `salloc` as described in the [NERSC docs](https://docs.nersc.gov/jobs/).
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
You can also choose to run only the array implementation, or only the Bricks implementation.
```bash
fft-gene-6d 5 100 a  # Run array layout only
fft-gene-6d 5 100 b  # Run bricks layout only
```

### Batch jobs

To run the batch jobs, we're going to first need to [build the slurm scripts](#building-the-slurm-scripts).

#### Building the slurm scripts

First, choose a directory you want to the builds and data to occur in.
Let's call that directory `$BRICKS_WORKSPACE`. For example, you could use the scratch directory.
Do this outside of the container, if you are using the shifter image.
```bash
export BRICKS_WORKSPACE="${SCRATCH}/bricks-benchmarks"
mkdir -p "${BRICKS_WORKSPACE}"
cd "${BRICKS_WORKSPACE}"
```

If you're using shifter, go ahead and start running the container and move to your workspace directory.
```bash
# Shifter build:
shifter --image=${BRICKS_SHIFTER_IMG} ${BRICKS_SHIFTER_ARGS} /bin/bash
export BRICKS_WORKSPACE="${SCRATCH}/bricks-benchmarks"
cd "${BRICKS_WORKSPACE}"
```

Now we need to build the slurm scripts. First, let's get the necessary files into our workspace.
```bash
cp -r "${bricklib_SRCDIR}/gene/slurm/"* .
````
Now exit the container.
```bash
# Shifter build:
exit
```
If you wish to be emailed when the jobs start, or to charge a specific account,
set the `BRICKS_ACCOUNT` or `BRICKS_EMAIL` environment variables.
If you are planning to use the shifter image, make sure the `BRICKS_SHIFTER_IMG`
variable is set as described [above](#setup-using-shifter-on-perlmutter).
```bash
# make jobs with no email, no account
make
# make jobs with email and account
BRICKS_ACCOUNT=<account> BRICKS_EMAIL=<email> make
```
This will create several files in the `generated-scripts` directory.
* `generated-scripts/single-stencil.h`: Run the `single-gene-6d` benchmark.
* `generated-scripts/fft.h`: Run the `fft-gene-6d` benchmark.

You can see customization options for the slurm scripts by running
```bash
python3 brick_shape_slurm_gen.py -h
python3 fft_slurm_gen.py -h
```

#### Submitting the jobs

If you are not using the shifter image,
make sure you've loaded the modules as in [`perlmutter_setup.sh`](https://github.com/benSepanski/bricklib/blob/sc-22-artifact/gene/docker/perlmutter_setup.sh).
Submitting these scripts will cause some cmake builds to occur.
Otherwise, use `exit` to exit the container (`sbatch` is not defined in the container).

Now we can submit the jobs. Make sure to run this from the `${BRICKS_WORKSPACE}` directory,
outside the container.
```bash
sbatch generated-scripts/single-stencil.h
sbatch generated-scripts/fft.h
```

#### Finding the results

All data will be stored in `.csv`s in  `${BRICKS_WORKSPACE}`.
* Single-device stencils: Stored in 3 `.csv`s:
  * `ncu_brick_shape_out.csv` Results from [NSight Compute](https://developer.nvidia.com/nsight-compute)
  * `brick_shape_out.csv` Results from our timing
  * `ptx_info_brick_shape.csv` Some results extracted from the `.ptx` files
* FFT: `fft_results`

You can plot the data using [RStudio](https://www.rstudio.com)
with the scripts included with our data used in the submission at
one of the author's [website](https://www.cs.utexas.edu/~bmsepan/projects/highDimensionalBricks/).
  
