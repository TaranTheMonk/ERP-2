from abc import ABC
from typing import Tuple


class BaseSolver(ABC):
    def solve(self) -> Tuple[float, float, float]:
        """
        :return: total reward, perfect solved, partial solved, computation time in seconds
        """
        pass
