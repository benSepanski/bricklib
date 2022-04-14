ARG UBUNTU_VERSION=20.04
ARG CUDA_ARCHITECTURE=80
ARG BUILDING_IMAGE_ON_PERLMUTTER=OFF
ARG NUM_THREADS=30

# -----------------------------------------------------------
# Build GTensor and Bricks with CUDA and MPI support
FROM nvidia/cuda:11.2.2-devel-ubuntu${UBUNTU_VERSION} AS buildwithcuda
ARG TARGETARCH
ARG CUDA_ARCHITECTURE
ARG BUILDING_IMAGE_ON_PERLMUTTER
ARG NUM_THREADS
SHELL ["/bin/bash", "-c"]
ENV HOME=/bricks/sc_22_submission

RUN apt-get update && apt-get install --yes \
         build-essential \
         curl \
         gfortran \
         git \
         python3.8 \
         wget

# Install CMake 3.23
WORKDIR ${HOME}
ENV CMAKE_VERSION=3.23.1
ENV CMAKE_DIR=${HOME}/cmake-${CMAKE_VERSION}
RUN if [[ "$TARGETARCH" == "arm64" ]] ; then \
        export CMAKE_ARCH="linux-aarch64" ; \
    else \
        export CMAKE_ARCH="linux-x86_64" ; \
    fi ; \
    cmake_file_name="cmake-${CMAKE_VERSION}-${CMAKE_ARCH}" ; \
    url="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${cmake_file_name}.tar.gz" ; \
    curl -L ${url} -o ${CMAKE_DIR}.tar.gz ; \
    tar -xzvf ${CMAKE_DIR}.tar.gz ; \
    mv "${cmake_file_name}" "${CMAKE_DIR}" ;
ENV PATH="${PATH}:${CMAKE_DIR}/bin"

# Install MPICH (https://docs.nersc.gov/development/shifter/how-to-use/#using-mpi-in-shifter)
ARG mpich=3.3
ARG mpich_prefix=mpich-$mpich

RUN \
    wget https://www.mpich.org/static/downloads/$mpich/$mpich_prefix.tar.gz && \
    tar xvzf $mpich_prefix.tar.gz                                           && \
    cd $mpich_prefix                                                        && \
    ./configure                                                             && \
    make -j ${NUM_THREADS}                                                  && \
    make install                                                            && \
    make clean                                                              && \
    cd ..                                                                   && \
    rm -rf $mpich_prefix

RUN /sbin/ldconfig

# download and build gtensor
RUN git clone https://github.com/wdmapp/gtensor.git
WORKDIR ${HOME}/gtensor 
RUN git reset --hard 41cf4fe26625f8d7ba2d0d3886a54ae6415a2017 
RUN cmake -S . -B build -DGTENSOR_DEVICE=cuda \
  -DCMAKE_INSTALL_PREFIX=bin \
  -DBUILD_TESTING=OFF \
  -DCMAKE_BUILD_TYPE=Release
RUN cmake --build build --target install
ENV gtensor_DIR=${HOME}/gtensor/bin

# download and build bricks
WORKDIR ${HOME}
RUN git clone https://github.com/benSepanski/bricklib.git # (While waiting for PR to work its way through bitbucket, pull from fork)
WORKDIR bricklib
RUN git checkout 5abff1121025a04858f4b85ec2260de435169b27 # TODO: Switch to tag once finalized
RUN cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX=bin \
  -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURE}" \
  -DGENE6D_USE_TYPES=OFF  \
  -DGENE6D_CUDA_AWARE=ON \
  -DPERLMUTTER=${BUILDING_IMAGE_ON_PERLMUTTER} `#Only switch to ON if building *image* on Perlmutter, not necessary to run container on Perlmutter` \
  -DCMAKE_CUDA_FLAGS="-lineinfo -gencode arch=compute_${CUDA_ARCHITECTURE},code=[sm_${CUDA_ARCHITECTURE},lto_${CUDA_ARCHITECTURE}]"
RUN cmake --build build --parallel

WORKDIR ${HOME}
ENTRYPOINT "cd ${HOME}/bricklib"
