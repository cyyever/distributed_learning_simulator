class Topology:
    def create_global_lock(self):
        raise NotImplementedError()

    @property
    def global_lock(self):
        raise NotImplementedError()
