import abc


class SRGRConverter(abc.ABC):

    @abc.abstractmethod
    def gr_to_rng(self, gr, azt):
        pass

    @abc.abstractmethod
    def rng_to_gr(self, gr, azt):
        pass
