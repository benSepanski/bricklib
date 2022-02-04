class MachineConfig:
    def __init__(self,
                 name=None,
                 gpus_per_node=None,
                 sockets_per_node=None,
                 cores_per_socket=None,
                 threads_per_core=None,
                 cuda_arch=None,
                 ):
        self.name = name
        self.gpus_per_node = gpus_per_node
        self.sockets_per_node = sockets_per_node
        self.cores_per_socket = cores_per_socket
        self.threads_per_core = threads_per_core
        self.cuda_arch = cuda_arch


machine_configurations = {
    "perlmutter": MachineConfig("perlmutter",
                                gpus_per_node=4,
                                sockets_per_node=1,
                                cores_per_socket=64,
                                threads_per_core=2,
                                cuda_arch=80,
                                ),
    "cori-gpu": MachineConfig("cori-gpu",
                              gpus_per_node=8,
                              sockets_per_node=2,
                              cores_per_socket=20,
                              threads_per_core=2,
                              cuda_arch=70)
}
