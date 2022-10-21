import time
from typing import Tuple, List, Iterable
from src.pkgs.sovlers.base_solver import BaseSolver
from src.pkgs.sovlers.mip_solver import MIPSolver
from src.pkgs.structs.task import Task
from src.pkgs.structs.worker import Worker


class BatchMIPSolver(BaseSolver):
    def __init__(self, n: int, workers: List[Worker], tasks: List[Task]):
        """
        @param n: solver splits the map to n * n squares for batching.
        @param workers:
        @param tasks:
        """
        self.n = n
        self.workers = workers
        self.tasks = tasks

    def solve(self) -> Tuple[float, float, float]:
        reward, solved = 0.0, 0.0
        start = time.time()
        for w, t in self.batching():
            if len(w) > 0 and len(t) > 0:
                _reward, _solved, _ = MIPSolver(w, t).solve()
                reward += _reward
                solved += _solved
        end = time.time()
        return reward, solved, end - start

    def batching(self) -> Iterable[Tuple[List[Worker], List[Task]]]:
        """
        random batch
        """
        min_lat = min([x.lat for x in self.workers] + [x.lat for x in self.tasks])
        min_lon = min([x.lon for x in self.workers] + [x.lon for x in self.tasks])
        max_lat = max([x.lat for x in self.workers] + [x.lat for x in self.tasks])
        max_lon = max([x.lon for x in self.workers] + [x.lon for x in self.tasks])

        lat_delta = (max_lat - min_lat) / self.n
        lon_delta = (max_lon - min_lon) / self.n

        for i in range(self.n):
            for j in range(self.n):
                w = list(filter(lambda x: (min_lat + i * lat_delta <= x.lat < min_lat + (i + 1) * lat_delta) and (
                            min_lon + j * lon_delta <= x.lon < min_lon + (j + 1) * lon_delta), self.workers))
                t = list(filter(lambda x: (min_lat + i * lat_delta <= x.lat < min_lat + (i + 1) * lat_delta) and (
                        min_lon + j * lon_delta <= x.lon < min_lon + (j + 1) * lon_delta), self.tasks))
                yield w, t
