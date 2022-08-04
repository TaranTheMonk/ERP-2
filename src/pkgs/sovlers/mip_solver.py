import time
from typing import Tuple, List
from pulp import (
    LpProblem,
    LpMaximize,
    LpVariable,
    LpBinary,
    lpSum,
    LpContinuous,
    lpDot,
    value,
    PULP_CBC_CMD,
    LpStatus, LpStatusInfeasible,
)
from src.pkgs.sovlers.base_solver import BaseSolver
from src.pkgs.structs.task import Task
from src.pkgs.structs.utils import travel_time, is_worker_available
from src.pkgs.structs.worker import Worker


class MIPSolver(BaseSolver):
    def __init__(self, workers: List[Worker], tasks: List[Task]):
        self.workers = workers
        self.tasks = tasks

    def solve(self) -> Tuple[float, float]:
        start = time.time()
        reward = self._mip_solve()
        end = time.time()
        return reward, end - start

    def _mip_solve(self) -> float:
        """
        Constants:
        s_i: stands for i-th task:
            s_i.maxR: max reward
            s_i.t_e: finish time
            s_i.d: deadline
            s_i.e: expected finish time
            s_i.pr: penalty rate
            s_i.wl: workload
            s_i.lat: latitude
            s_i.lon: longitude

        w_i: stands for j-th worker:
            w_i.lat: latitude
            w_i.lon: longitude
            w_i.min_lat: min latitude
            w_i.min_lon: min longitude
            w_i.max_lat: max latitude
            w_i.max_lon: min longitude
            w_i.v: velocity

        t_ij: stands for the travel time between i-th worker and j-th task.

        Variables:
        A_ij: stands for assign i-th worker to j-th task, binary.
        s_i.t_e: stands for i-th task's finish time, continuous.
        r_i: stands for i-th task's reward
        delta_ij: used for conditional constraints, binary.
        h_ij: used for linearizing the non-linear constraint, continues.



        maximize sum(r_i)
        s.t.:
            sum(A_ij) <= 1, for i in workers. Each worker can only be assigned to one task.

            # geo constraints
            A_ij * w_i.min_lat <= A_ij * s_j.lat <= A_ij * w_i.max_lat, if assigned, should be in the range.
            A_ij * w_i.min_lon <= A_ij * s_j.lat <= A_ij * w_i.max_lon, if assigned, should be in the range.
            =====>
            A_ij == 0, if any of the two above are violated.

            # non-linear constraint
            s_j.t_e == (sum(A_ij * t_ij) + s_j.wl) / sum(A_ij)
            s_j.t_e * sum(A_ij) == (sum(A_ij * t_ij) + s_j.wl)
            =====> linearize =====>
            h_ij >= A_ij * M
            h_ij <= A_ij * M
            h_ij >= s_j.t_e - (1 - A_ij) * M
            h_ij <= s_j.t_e + (1 - A_ij) * M

            if A_ij = 0, 0 <= h_ij <= 0.
            if A_ij = 1, s_j.t_e <= h_ij <= s_j.t_e
            =====> linearize =====>
            sum(h_ij) == sum(A_ij * t_ij) + s_j.wl

            # if-else constraints
            if s_i.t_e < s_i.d:
                r_i = s_i.maxR - (s_i.t_e - s_i.e) * s_i.pr
            else:
                r_i = 0
            =====>
            # if s_i.t_e >= s_i.d, delta_i must be 1.
            # if s_i.t_e < s_i.d, delta_i must be 0.
            s_i.t_e >= s_i.d - M * (1 - delta_i)
            s_i.t_e <= s_i.d - 0.001 + M * delta_i

            # if delta_i is 0, r_i = (s_i.maxR - (s_i.t_e - s_i.e) * s_i.pr)
            # if delta_i = 1, r_i = 0
            r_i <= (s_i.maxR - (s_i.t_e - s_i.e) * s_i.pr) + M * delta_i
            r_i >= (s_i.maxR - (s_i.t_e - s_i.e) * s_i.pr) - M * delta_i
            r_i <= M * (1 - delta_i)
            r_i >= -M * (1 - delta_i)
            r_i <= s_i.maxR, in case finish before the expected time
        """
        # constants
        M = 10e3

        # compute travel time
        t = [[0] * len(self.tasks) for _ in range(len(self.workers))]
        reverse_t = [[0] * len(self.workers) for _ in range(len(self.tasks))]
        for i in range(len(self.workers)):
            for j in range(len(self.tasks)):
                _ = travel_time(
                        self.workers[i].lat,
                        self.workers[i].lon,
                        self.tasks[j].lat,
                        self.tasks[j].lon,
                        self.workers[i].velocity,
                    )
                t[i][j] = _
                reverse_t[j][i] = _

        # create a problem
        prob = LpProblem("my_problem", LpMaximize)

        # create variables
        r = list()
        for i in range(len(self.tasks)):
            r.append(
                LpVariable(
                    f"r_{i}",
                    lowBound=0.0,
                    upBound=self.tasks[i].reward,
                    cat=LpContinuous,
                )
            )

        a = [[0] * len(self.tasks) for _ in range(len(self.workers))]
        reverse_a = [[0] * len(self.workers) for _ in range(len(self.tasks))]
        for i in range(len(self.workers)):
            for j in range(len(self.tasks)):
                _ = LpVariable(f"A_{i}_{j}", cat=LpBinary)
                a[i][j] = _
                reverse_a[j][i] = _

        t_e = list()
        for i in range(len(self.tasks)):
            t_e.append(LpVariable(f"t_e_{i}", lowBound=0.0, cat=LpContinuous))

        delta = list()
        beta = list()
        for i in range(len(self.tasks)):
            delta.append(LpVariable(f"delta_{i}", cat=LpBinary))
            beta.append(LpVariable(f"beta_{i}", cat=LpBinary))

        h = [[0] * len(self.tasks) for _ in range(len(self.workers))]
        reverse_h = [[0] * len(self.workers) for _ in range(len(self.tasks))]
        for i in range(len(self.workers)):
            for j in range(len(self.tasks)):
                _ = LpVariable(f"h_{i}_{j}", lowBound=0.0, cat=LpContinuous)
                h[i][j] = _
                reverse_h[j][i] = _

        # add constraints
        for i in range(len(self.workers)):
            for j in range(len(self.tasks)):
                if not is_worker_available(self.workers[i], self.tasks[j]):
                    prob += a[i][j] == 0

                prob += h[i][j] <= a[i][j] * M
                prob += h[i][j] >= -a[i][j] * M
                prob += h[i][j] <= t_e[j] + (1 - a[i][j]) * M
                prob += h[i][j] >= t_e[j] - (1 - a[i][j]) * M

            prob += lpSum(a[i]) <= 1

        for i in range(len(self.tasks)):
            task = self.tasks[i]
            # beta = 1 if lpSum(reverse_a[i]) = 1
            # beta = 0 if lpSum(reverse_a[i]) = 0
            prob += lpSum(reverse_a[i]) >= 1 - M * (1 - beta[i])
            prob += lpSum(reverse_a[i]) <= 0 + M * beta[i]

            # if beta = 0, sum(h) = 0 => t_e >= deadline
            # if beta = 1, sum(h) = sum(a * t) + wl >= t_e = h
            prob += lpSum(reverse_h[i]) >= lpDot(reverse_a[i], reverse_t[i]) + task.workload - M * (1 - beta[i])
            prob += lpSum(reverse_h[i]) <= lpDot(reverse_a[i], reverse_t[i]) + task.workload + M * (1 - beta[i])
            prob += lpSum(reverse_h[i]) >= 0 - M * beta[i]
            prob += lpSum(reverse_h[i]) <= 0 + M * beta[i]

            # if beta = 0, t_e >= deadline
            prob += t_e[i] >= task.deadline + 1.0 - M * beta[i]
            prob += t_e[i] <= task.deadline + 1.0 + M * beta[i]

            prob += t_e[i] >= task.deadline - M * (1 - delta[i])
            prob += t_e[i] <= task.deadline - 0.001 + M * delta[i]

            prob += (
                r[i]
                <= task.reward
                - (t_e[i] - task.expected_time) * task.penalty_rate
                + M * delta[i]
            )
            prob += (
                r[i]
                >= task.reward
                - (t_e[i] - task.expected_time) * task.penalty_rate
                - M * delta[i]
            )
            prob += r[i] <= M * (1 - delta[i])
            prob += r[i] >= -M * (1 - delta[i])

        # add objective
        prob += lpSum(r)

        # solve
        status = prob.solve(PULP_CBC_CMD(msg=False, timeLimit=10))

        # debug
        # task = self.tasks[0]
        # print(task.lat, task.lon)
        # for w in self.workers:
        #     print(w.min_lat, w.max_lat, w.min_lon, w.max_lon)
        #
        # print(f"woker assigned: {value(a[0][0]), value(a[1][0])}")
        # print(f"task assigned: {value(beta[0])}")
        # print(f"max reward: {task.reward}")
        # print(f"worker time: {value(h[0][0]), value(h[1][0])}")
        # print(f"finish time: {value(t_e[0])}")
        # print(f"expected time: {task.expected_time}")
        # print(f"deadline: {task.deadline}")
        # print(f"delta: {value(delta[0])}")
        # print(f"true reward: {value(r[0])}")
        # print("\n")

        if status is LpStatusInfeasible:
            return 0.0
        else:
            return sum(value(r_i) for r_i in r)
