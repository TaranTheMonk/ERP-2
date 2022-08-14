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
    pathlib.Path(__file__).parent, "../resources/processed_data"
)


def solve(instance_id: int, instance_size: int):
    # results
    r_1, solved_1, t_1 = [], [], []
    r_2, solved_2, t_2 = [], [], []
    r_3, solved_3, t_3 = [], [], []

    tmp = {"instance_size": instance_size}

    for i in range(5):
        # read workers
        workers = list()
        for w_id, pd_ser in pd.read_csv(
            os.path.join(RESOURCE_PATH, f"worker_{instance_id}/workers{i}.csv")
        ).iterrows():
            workers.append(Worker.from_pd_series(w_id, pd_ser))

        # read tasks
        tasks = list()
        for t_id, pd_ser in pd.read_csv(
            os.path.join(RESOURCE_PATH, f"task_{instance_id}/tasks{i}.csv")
        ).iterrows():
            tasks.append(Task.from_pd_series(t_id, pd_ser))

        # solver 1
        greed_by_reward_solver = GreedyByRewardSolver(
            workers=workers[:instance_size], tasks=tasks[:instance_size]
        )
        _r, _solved, _t = greed_by_reward_solver.solve()
        r_1.append(_r)
        solved_1.append(_solved)
        t_1.append(_t)

        # solver 2
        greed_by_reward_per_workload_solver = GreedyByRewardPerWorkloadSolver(
            workers=workers[:instance_size], tasks=tasks[:instance_size]
        )
        _r, _solved, _t = greed_by_reward_per_workload_solver.solve()
        r_2.append(_r)
        solved_2.append(_solved)
        t_2.append(_t)

        # solver 3
        mip_solver = MIPSolver(
            workers=workers[:instance_size], tasks=tasks[:instance_size]
        )
        _r, _solved, _t = mip_solver.solve()
        if _r >= 0:
            r_3.append(_r)
            solved_3.append(_solved)
            t_3.append(_t)

    print("")
    print(f"instance size: {instance_size}")
    print("#########################")
    print(f"greedy by reward solver:")
    print(f"avg_reward: {mean(r_1)}, avg_time: {mean(t_1)}, avg_solved: {mean(solved_1)}")
    print("#########################")
    print(f"greedy by reward per workload solver:")
    print(f"avg_reward: {mean(r_2)}, avg_time: {mean(t_2)}, avg_solved: {mean(solved_2)}")
    print("#########################")
    print(f"MIP solver:")
    print(f"avg_reward: {mean(r_3)}, avg_time: {mean(t_3)}, avg_solved: {mean(solved_3)}")
    print(f"solved: {len(r_3)}")
    print("#########################")

    tmp["r1"] = mean(r_1)
    tmp["solved1"] = mean(solved_1)
    tmp["t1"] = mean(t_1)

    tmp["r2"] = mean(r_2)
    tmp["solved2"] = mean(solved_2)
    tmp["t2"] = mean(t_2)

    tmp["r3"] = mean(r_3)
    tmp["solved3"] = mean(solved_3)
    tmp["t3"] = mean(t_3)

    return tmp


if __name__ == "__main__":
    res = {
        "instance_size": list(),
        "r1": list(),
        "solved1": list(),
        "t1": list(),
        "r2": list(),
        "solved2": list(),
        "t2": list(),
        "r3": list(),
        "solved3": list(),
        "t3": list()
    }

    for x in range(10, 110, 10):
        _res = solve(instance_id=100, instance_size=x)
        for k in res.keys():
            res[k].append(_res[k])

    for x in range(110, 210, 10):
        _res = solve(instance_id=200, instance_size=x)
        for k in res.keys():
            res[k].append(_res[k])

    pd.DataFrame(res).to_csv("result.csv", index=False)
