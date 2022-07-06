import os
import pandas as pd
import pathlib

from src.pkgs.sovlers.greedy_by_reward_per_workload_solver import (
    GreedyByRewardPerWorkloadSolver,
)
from src.pkgs.sovlers.greedy_by_reward_solver import GreedyByRewardSolver
from src.pkgs.structs.task import Task
from src.pkgs.structs.worker import Worker

RESOURCE_PATH = os.path.join(
    pathlib.Path(__file__).parent, "../.resources/processed_data"
)


def solve(instance_size: int):
    # total reward and cost
    r_1, t_1 = 0.0, 0.0
    r_2, t_2 = 0.0, 0.0

    for i in range(30):
        # read workers
        workers = list()
        for w_id, pd_ser in pd.read_csv(
            os.path.join(RESOURCE_PATH, f"worker_{instance_size}/workers{i}.csv")
        ).iterrows():
            workers.append(Worker.from_pd_series(w_id, pd_ser))

        # read tasks
        tasks = list()
        for t_id, pd_ser in pd.read_csv(
            os.path.join(RESOURCE_PATH, f"task_{instance_size}/tasks{i}.csv")
        ).iterrows():
            tasks.append(Task.from_pd_series(t_id, pd_ser))

        # solver 1
        greed_by_reward_solver = GreedyByRewardSolver(workers=workers, tasks=tasks)
        _r, _t = greed_by_reward_solver.solve()
        r_1 += _r
        t_1 += _t

        # solver 2
        greed_by_reward_per_workload_solver = GreedyByRewardPerWorkloadSolver(
            workers=workers, tasks=tasks
        )
        _r, _t = greed_by_reward_per_workload_solver.solve()
        r_2 += _r
        t_2 += _t

    print(f"instance size: {instance_size}")
    print(f"greedy by reward solver:")
    print(f"avg_reward: {r_1 / 30}, avg_time: {t_1 / 30}")
    print("\n")
    print(f"greedy by reward per workload solver:")
    print(f"avg_reward: {r_2 / 30}, avg_time: {t_2 / 30}")
    print("\n")


if __name__ == "__main__":
    for x in range(100, 600, 100):
        solve(x)
