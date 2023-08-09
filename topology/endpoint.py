from .topology import Topology


class Endpoint:
    def __init__(self, topology: Topology):
        self._topology: Topology = topology

    # @property
    # def topology_lock(self):
    #     return self._topology.global_lock
