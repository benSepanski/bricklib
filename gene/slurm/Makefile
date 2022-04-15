PYTHONPATH=$(shell pwd)/slurmpy:$(shell printenv PYTHONPATH)
ifdef BRICKS_ACCOUNT
  BRICKS_ACCOUNT=-A $(shell printenv ACCOUNT)
else
  BRICKS_ACCOUNT=
endif
ifdef BRICKS_EMAIL
  BRICKS_EMAIL=-e $(shell printenv BRICKS_EMAIL)
else
  BRICKS_EMAIL=
endif
ifdef BRICKS_SHIFTER_IMG
  BRICKS_SHIFTER_IMG=-i $(shell printenv BRICKS_SHIFTER_IMG)
  bricklib_SRCDIR=/bricks/sc_22_submission/bricklib
  gtensor_DIR=/bricks/sc_22_submission/deps/gtensor/bin
else
  BRICKS_SHIFTER_IMG=
  ifndef bricklib_SRCDIR
    $(error bricklib_SRCDIR is not defined)
  else
    bricklib_SRCDIR = $(shell printenv bricklib_SRCDIR)
  endif
endif
ARGS=$(bricklib_SRCDIR) -M perlmutter $(BRICKS_ACCOUNT) $(BRICKS_EMAIL) $(BRICKS_SHIFTER_IMG)
MPI_ARGS=$(ARGS) --always-cuda-aware

.PHONY: mpi-stencil mpi-stencil-strong mpi-stencil-weak
all: generated-scripts/single-stencil.sh generated-scripts/fft.h mpi-stencil

generated-scripts/single-stencil.sh:
	mkdir -p generated-scripts
	python3 brick_shape_slurm_gen.py $(ARGS) > generated-scripts/single-stencil.sh
	chmod +x generated-scripts/single-stencil.sh

generated-scripts/fft.h:
	python3 fft_slurm_gen.py $(ARGS) > generated-scripts/fft.sh
	chmod +x generated-scripts/fft.sh

mpi-stencil-strong:
	python3 mpi_slurm_gen.py 4 $(MPI_ARGS)
	python3 mpi_slurm_gen.py 8 $(MPI_ARGS)
	python3 mpi_slurm_gen.py 16 $(MPI_ARGS)
	python3 mpi_slurm_gen.py 32 $(MPI_ARGS)
	python3 mpi_slurm_gen.py 64 $(MPI_ARGS)
	python3 mpi_slurm_gen.py 128 $(MPI_ARGS)

mpi-stencil-weak:
	python3 mpi_slurm_gen.py 4 -d 70,16,24,48,32,2 -J mpi_4_weak_exascale_gene_domain $(MPI_ARGS)
	python3 mpi_slurm_gen.py 8 -d 280,32,24,48,32,2 -J mpi_8_weak_exascale_gene_domain $(MPI_ARGS)
	python3 mpi_slurm_gen.py 16 -d 560,32,24,48,32,2 -J mpi_16_weak_exascale_gene_domain $(MPI_ARGS)
	python3 mpi_slurm_gen.py 32 -d 560,32,24,96,32,2 -J mpi_32_weak_exascale_gene_domain $(MPI_ARGS)
	python3 mpi_slurm_gen.py 64 -d 560,32,24,96,64,2 -J mpi_64_weak_exascale_gene_domain $(MPI_ARGS)
	python3 mpi_slurm_gen.py 128 -d 1120,32,24,96,64,2 -J mpi_128_weak_exascale_gene_domain $(MPI_ARGS)


mpi-stencil: mpi-stencil-strong mpi-stencil-weak

