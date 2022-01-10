for use_types in ON OFF ; do
    for cuda_aware in ON OFF ; do
        dir_name="perlmutter_use_types_${use_types}_cuda_aware_${cuda_aware}"
        if [ ! -d ${dir_name} ] ; then
            mkdir ${dir_name}
        fi
        cd ${dir_name} ;
        cmake ../../../ \
            -DCMAKE_CUDA_ARCHITECTURES="80" \
            -DCMAKE_INSTALL_PREFIX=bin \
            -DGENE6D_USE_TYPES=${use_types} \
            -DGENE6D_CUDA_AWARE=${cuda_aware} \
            -DCMAKE_CUDA_FLAGS="-lineinfo -gencode arch=compute_80,code=[sm_80,lto_80]" \
            -DCMAKE_BUILD_TYPE=Release \
            -DPERLMUTTER=ON ;
        make -j 20;
        cd .. ;
    done;
done;
