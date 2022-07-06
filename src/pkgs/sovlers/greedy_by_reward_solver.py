import time
from typing import List, Tuple

from src.pkgs.sovlers.base_solver import BaseSolver
from src.pkgs.structs.task import Task
from src.pkgs.structs.utils import get_sorted_available_workers, get_finish_time
from src.pkgs.structs.worker import Worker


class GreedyByRewardSolver(BaseSolver):
    """
    1. finish tasks from high reward to low.
    2. select workers from close to far.
    3. use worker set that maximize the reward.
    4. try finish as more tasks as possible.
    """

    def __init__(self, workers: List[Worker], tasks: List[Task]):
        self.workers = workers
        self.tasks = tasks

    def greedy_solve(self) -> float:
        # desc sort tasks by reward
        self.tasks.sort(key=lambda x: x.reward, reverse=True)

        # solve
        workers_set = set(self.workers)
        assigned_workers = set()
        reward = 0.0
        for t in self.tasks:
            best_reward = 0.0
            best_workers = list()

            # get available_workers
            available_workers = get_sorted_available_workers(
                workers_set - assigned_workers, t
            )

            # no available workers
            if len(available_workers) == 0:
                continue

            # select workers from close to from
            for i in range(1, len(available_workers)):
                finish_time = get_finish_time(available_workers[:i], t)

                if finish_time >= t.deadline:
                    _r = 0.0
                elif finish_time <= t.expected_time:
                    _r = t.reward
                else:
                    _r = t.reward - t.penalty_rate * (finish_time - t.expected_time)

                if _r > best_reward:
                    best_reward = _r
                    best_workers = available_workers[:i]

            # update reward and assigned_workers
            reward += best_reward
            for w in best_workers:
                assigned_workers.add(w)

        return reward

    def solve(self) -> Tuple[float, float]:
        start = time.time()
        reward = self.greedy_solve()
        end = time.time()
        return reward, end - start
