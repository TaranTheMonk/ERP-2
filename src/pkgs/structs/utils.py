import math
from typing import List, Iterable

from src.pkgs.structs.task import Task
from src.pkgs.structs.worker import Worker


def travel_time(
    lat_1: float, lon_1: float, lat_2: float, lon_2: float, v: float
) -> float:
    return math.sqrt((lat_1 - lat_2) ** 2 + (lon_1 - lon_2) ** 2) / v


def is_worker_available(w: Worker, t: Task) -> bool:
    """
    give a worker and a task, check if the worker is
    available to the task
    :param w:
    :param t:
    :return:
    """
    # out of workers range
    if t.lat < w.min_lat or t.lat > w.max_lat:
        return False
    if t.lon < w.min_lon or t.lon > w.max_lon:
        return False

    # no contribution
    if travel_time(w.lat, w.lon, t.lat, t.lon, w.velocity) > t.deadline:
        return False

    return True


def get_sorted_available_workers(workers: Iterable[Worker], t: Task) -> List[Worker]:
    """
    given multiple workers and a task, return a list of
    available workers asc sorted by travel_time
    :param workers:
    :param t:
    :return:
    """
    ret = list()
    for w in workers:
        if is_worker_available(w, t):
            ret.append(w)

    ret.sort(key=lambda x: travel_time(x.lat, x.lon, t.lat, t.lon, x.velocity))
    return ret


def get_finish_time(workers: Iterable[Worker], t: Task) -> float:
    """
    given multiple workers and a task, compute the finish time.

    How?
    1. Obviously,
        finish_time = worker's travel_time + worker's work_time
    2. sum over all workers for the right side,
        finish_time * worker_num = total_travel_time + task's work_load
    :param workers:
    :param t:
    :return:
    """
    total_travel = 0.0
    w_cnt = 0

    for w in workers:
        travel_time(w.lat, w.lon, t.lat, t.lon, w.velocity)
        w_cnt += 1

    return (total_travel + t.workload) / w_cnt
