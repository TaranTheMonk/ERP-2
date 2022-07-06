from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class Task:
    id: int
    lat: float
    lon: float
    deadline: float
    workload: float
    expected_time: float
    penalty_rate: float
    reward: float

    @classmethod
    def from_pd_series(cls, t_id: int, ser: pd.Series) -> "Task":
        return cls(
            id=t_id,
            lat=float(ser.lat),
            lon=float(ser.lon),
            deadline=float(ser.deadline),
            workload=float(ser.workload),
            expected_time=float(ser.expected_time),
            penalty_rate=float(ser.penalty_rate),
            reward=float(ser.reward),
        )
