import abc


class Solver(abc.ABC):
    @abc.abstractmethod
    def solve(self, A, b):
        pass
