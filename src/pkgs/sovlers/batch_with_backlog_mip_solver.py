import random
import time
from typing import Tuple, List, Iterable
from src.pkgs.sovlers.base_solver import BaseSolver
from src.pkgs.sovlers.mip_solver import MIPSolver
from src.pkgs.structs.task import Task
from src.pkgs.structs.worker import Worker


class BatchWithBacklogMIPSolver(BaseSolver):
    def __init__(self, n: int, backlog_size: int, workers: List[Worker], tasks: List[Task]):
        """
        @param n: solver splits the map to n * n squares for batching.
        @param backlog_size: backlog size
        @param workers:
        @param tasks:
        """
        self.n = n
        self.backlog_size = backlog_size
        self.workers = workers
        self.tasks = tasks

    def solve(self) -> Tuple[float, float, float]:
        reward, solved = 0.0, 0.0
        start = time.time()
        backlog_w = list()
        backlog_t = list()
        for w, t in self.batching():
            if len(w) > 0 and len(t) > 0:
                w += backlog_w
                t += backlog_t

                _reward, _solved, _, assignments = MIPSolver(w, t).solve()
                reward += _reward
                solved += _solved

                # update backlog
                w_id_set = set()
                t_id_set = set()
                for w_id, t_id in assignments:
                    w_id_set.add(w_id)
                    t_id_set.add(t_id)
                backlog_w = list(filter(lambda x: x.id not in w_id_set, w))
                backlog_t = list(filter(lambda x: x.id not in t_id_set, t))
                backlog_w = random.sample(backlog_w, k=min(self.backlog_size, len(backlog_w)))
                backlog_t = random.sample(backlog_t, k=min(self.backlog_size, len(backlog_t)))

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
