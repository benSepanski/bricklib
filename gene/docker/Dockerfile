ARG UBUNTU_VERSION=20.04
ARG CUDA_ARCHITECTURE=80
ARG BUILDING_IMAGE_ON_PERLMUTTER=OFF
ARG NUM_THREADS=30

# -----------------------------------------------------------
# Build GTensor and Bricks with CUDA and MPI support
FROM nvidia/cuda:11.4.1-devel-ubuntu${UBUNTU_VERSION} AS buildwithcuda
ARG TARGETARCH
ARG CUDA_ARCHITECTURE
ARG BUILDING_IMAGE_ON_PERLMUTTER
ARG NUM_THREADS
SHELL ["/bin/bash", "-c"]
ENV HOME=/bricks/sc_22_submission
ENV BRICKS_DEPS=${HOME}/deps
ENV CUDA_ARCHITECTURE=$CUDA_ARCHITECTURE

RUN apt-get update && apt-get install --yes \
         build-essential \
         curl \
         gcc-10 \
         gfortran \
         git \
         g++-10 \
         python3.8 \
         vim \
         wget

RUN for program in gcc g++ ; do \
        update-alternatives --install /usr/bin/${program} ${program} /usr/bin/${program}-10 10 ; \
        update-alternatives --set ${program} /usr/bin/${program}-10 ; \
    done ; \
    ln -sfn /usr/bin/cpp-10 /usr/bin/cpp ; \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 38 ; \
    update-alternatives --set python3 /usr/bin/python3.8 ; \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.8 38 ; \
    update-alternatives --set python /usr/bin/python3.8 ;
ENV CC=gcc
ENV CXX=g++

# Based on running the following on perlmutter on April 14, 2022
# g++ -march=native -E -v - </dev/null 2>&1 | grep cc1
# taken from https://stackoverflow.com/questions/5470257/how-to-see-which-flags-march-native-will-activate
ENV PERLMUTTER_CXX_FLAGS="-march=znver3 -mmmx -mno-3dnow -msse -msse2 -msse3 -mssse3 -msse4a -mcx16 -msahf -mmovbe -maes -msha -mpclmul -mpopcnt -mabm -mno-lwp -mfma -mno-fma4 -mno-xop -mbmi -mno-sgx -mbmi2 -mno-pconfig -mwbnoinvd -mno-tbm -mavx -mavx2 -msse4.2 -msse4.1 -mlzcnt -mno-rtm -mno-hle -mrdrnd -mf16c -mfsgsbase -mrdseed -mprfchw -madx -mfxsr -mxsave -mxsaveopt -mno-avx512f -mno-avx512er -mno-avx512cd -mno-avx512pf -mno-prefetchwt1 -mclflushopt -mxsavec -mxsaves -mno-avx512dq -mno-avx512bw -mno-avx512vl -mno-avx512ifma -mno-avx512vbmi -mno-avx5124fmaps -mno-avx5124vnniw -mclwb -mmwaitx -mclzero -mpku -mrdpid -mno-gfni -mshstk -mno-avx512vbmi2 -mno-avx512vnni -mvaes -mvpclmulqdq -mno-avx512bitalg -mno-avx512vpopcntdq -mno-movdiri -mno-movdir64b -mno-waitpkg -mno-cldemote -mno-ptwrite -mno-avx512bf16 -mno-enqcmd -mno-avx512vp2intersect --param l1-cache-size=32 --param l1-cache-line-size=64 --param l2-cache-size=512 -mtune=znver3"
ENV CXXFLAGS="${PERLMUTTER_CXX_FLAGS}"

# Install CMake 3.23
WORKDIR ${BRICKS_DEPS}
ENV CMAKE_VERSION=3.23.1
ENV CMAKE_DIR=${BRICKS_DEPS}/cmake-${CMAKE_VERSION}
RUN if [[ "$TARGETARCH" == "arm64" ]] ; then \
        export CMAKE_ARCH="linux-aarch64" ; \
    else \
        export CMAKE_ARCH="linux-x86_64" `# FIXME: this should be an elif`; \
    fi ; \
    cmake_file_name="cmake-${CMAKE_VERSION}-${CMAKE_ARCH}" ; \
    url="https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/${cmake_file_name}.tar.gz" ; \
    curl -L "${url}" -o "${CMAKE_DIR}.tar.gz" ; \
    tar -xzvf "${CMAKE_DIR}.tar.gz" ; \
    mv "${cmake_file_name}" "${CMAKE_DIR}" ; \
    rm "${CMAKE_DIR}.tar.gz"
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
    rm -rf $mpich_prefix $mpich_prefix.tar.gz

RUN /sbin/ldconfig

# download and build gtensor
RUN git clone https://github.com/wdmapp/gtensor.git && \
    cd gtensor && \
    git reset --hard 455afd5839243cca2fa59b129fa14deff7bf038a
WORKDIR ${BRICKS_DEPS}/gtensor 
RUN cmake -S . -B build \
        -DGTENSOR_DEVICE=cuda \
        -DCMAKE_INSTALL_PREFIX=bin \
        -DBUILD_TESTING=OFF \
        -DCMAKE_BUILD_TYPE=Release ; \
    cmake --build build --target install
ENV gtensor_DIR=${BRICKS_DEPS}/gtensor/bin

# Get OpenCL-hpp headers
WORKDIR ${BRICKS_DEPS}
RUN git clone https://github.com/KhronosGroup/OpenCL-CLHPP.git && \
    cd OpenCL-CLHPP && \
    git checkout tags/v2.0.16

# download and build bricks
WORKDIR ${HOME}
RUN git clone https://github.com/benSepanski/bricklib.git `# (While waiting for PR to work its way through bitbucket, pull from fork)` && \
    cd bricklib && \
    git checkout tags/sc22_artifact1.2
WORKDIR bricklib
RUN cmake -S . -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX=bin \
        -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHITECTURE}" \
        -DGENE6D_USE_TYPES=OFF  \
        -DGENE6D_CUDA_AWARE=ON \
        -DPERLMUTTER=${BUILDING_IMAGE_ON_PERLMUTTER} `#Only switch to ON if building *image* on Perlmutter, not necessary to run container on Perlmutter` \
        -DCMAKE_CUDA_FLAGS="-lineinfo -gencode arch=compute_${CUDA_ARCHITECTURE},code=[sm_${CUDA_ARCHITECTURE},lto_${CUDA_ARCHITECTURE}]" \
        -DCMAKE_CXX_FLAGS="${PERLMUTTER_CXX_FLAGS} -I/${BRICKS_DEPS}/OpenCL-CLHPP/include" \
        -DCMAKE_CUDA_HOST_COMPILER=g++ && \
    cmake --build build --parallel
# Put gene executables on path
ENV PATH=${HOME}/bricklib/build/gene:${PATH}
# put slurmpy on pythonpath for batch job generation
ENV PYTHONPATH="${HOME}/bricklib/gene/slurm/slurmpy:${PYTHONPATH}"
# Make sure to pre-load necessary librariesfor CUDA-Aware MPI (https://docs.nersc.gov/development/shifter/gpus/#shifter-cuda-mpich-module)
ENV LD_PRELOAD="/opt/udiImage/modules/cuda-mpich/lib64/libmpi_gtl_cuda.so.0 ${LD_PRELOAD}"
ENV MPICH_GPU_SUPPORT_ENABLED=1

ENTRYPOINT /bin/bash --init-file <(echo "cd ${SCRATCH}; if [[ ! -L ${SCRATCH}/bricklib ]] ; then ln -s ${HOME}/bricklib ; fi ; export bricklib_SRCDIR=${SCRATCH}/bricklib")
