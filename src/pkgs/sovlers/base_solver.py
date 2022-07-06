from abc import ABC
from typing import Tuple


class BaseSolver(ABC):
    def solve(self) -> Tuple[float, float]:
        """
        :return: total reward, computation time in seconds
        """
        pass
