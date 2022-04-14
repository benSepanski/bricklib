# GENE Benchmarks

FIXME: DO I NEED TO DESCRIBE CODE STRUCTURE?

This directory contains benchmarks based on the [GENE code](http://genecode.org)
and the [GTensor](https://github.com/wdmapp/gtensor) project used
to port GENE to GPU 
(see [Toward exascale whole-device modeling of fusion devices: Porting the GENE gyrokinetic microturbulence code to GPU](https://aip.scitation.org/doi/10.1063/5.0046327)
 in Physics of Plasmas, 2021).

## Setup

For our SC 2022 submission, we have automated the setup using [Docker](https://www.docker.com)
and [Shifter](https://docs.nersc.gov/development/shifter/) through NERSC.
The Dockerfile is located in the `docker` directory, if you wish to see the setup for yourself.
To replicate the results, please follow [the Shifter setup instructions](#setup-using-shifter-on-perlmutter)
below. Then, follow the instructions in (#running-the-benchmarks) below.

### Setup using Shifter on Perlmutter

These instructions are used to set up the results on the [NERSC Perlmutter cluster](https://docs.nersc.gov/systems/perlmutter/).
If you do not have access to perlmutter,
you can apply as described on [NERSC's website](https://docs.nersc.gov/systems/perlmutter/#access).

Begin by logging into perlmutter.
```bash
ssh perlmutter
```
Next, use `shifter` to set up the environment.
First, pull the image down from DockerHub.
```bash
shifterimg pull -v bensepanski/2022_sc_bricks:0.1  # FIXME: set tag
```
Then, start the container and move to the experiment directory.
```bash
shifter --image=bensepanski/2022_sc_bricks:0.1 \
        --module=gpu \
        --module=cuda-mpich
cd /bricks/sc_22_submission/bricklib/gene/slurm
```

### Manual setup

FIXME: DESCRIBE MANUAL SETUP HERE

## Running the benchmarks

FIXME: DESCRIBE WHAT TO DO