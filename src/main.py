import os
from statistics import mean
import pandas as pd
import pathlib

from src.pkgs.sovlers.greedy_by_reward_per_workload_solver import (
    GreedyByRewardPerWorkloadSolver,
)
from src.pkgs.sovlers.greedy_by_reward_solver import GreedyByRewardSolver
from src.pkgs.sovlers.mip_solver import MIPSolver
from src.pkgs.structs.task import Task
from src.pkgs.structs.worker import Worker

RESOURCE_PATH = os.path.join(
    pathlib.Path(__file__).parent, "../.resources/processed_data"
)


def solve(instance_size: int):
    # total reward and cost
    r_1, t_1 = [], []
    r_2, t_2 = [], []
    r_3, t_3 = [], []

    for i in range(4, 5):
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

        # # solver 1
        # greed_by_reward_solver = GreedyByRewardSolver(workers=workers, tasks=tasks)
        # _r, _t = greed_by_reward_solver.solve()
        # r_1.append(_r)
        # t_1.append(_t)
        #
        # solver 2
        greed_by_reward_per_workload_solver = GreedyByRewardPerWorkloadSolver(
            workers=workers[:50], tasks=tasks[:50]
        )
        _r, _t = greed_by_reward_per_workload_solver.solve()
        print(_r)
        r_2.append(_r)
        t_2.append(_t)

        # solver 3
        mip_solver = MIPSolver(
            workers=workers[:50], tasks=tasks[:50]
        )
        _r, _t = mip_solver.solve()
        print(_r)
        r_3.append(_r)
        t_3.append(_t)

    print("\n")
    print(f"instance size: {instance_size}")
    print("#########################")
    # print(f"greedy by reward solver:")
    # print(f"avg_reward: {mean(r_1)}, avg_time: {mean(t_1)}")
    # print("\n")
    # print(f"greedy by reward per workload solver:")
    # print(f"avg_reward: {mean(r_2)}, avg_time: {mean(t_2)}")
    # print("\n")
    print(f"MIP solver:")
    print(f"avg_reward: {mean(r_3)}, avg_time: {mean(t_3)}")
    print("\n")


if __name__ == "__main__":
    for x in range(100, 200, 100):
        solve(x)
