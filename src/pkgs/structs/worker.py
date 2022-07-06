from dataclasses import dataclass
import pandas as pd


@dataclass(frozen=True)
class Worker:
    id: int
    lat: float
    lon: float
    min_lat: float
    min_lon: float
    max_lat: float
    max_lon: float
    velocity: float

    @classmethod
    def from_pd_series(cls, w_id: int, ser: pd.Series) -> "Worker":
        return cls(
            id=w_id,
            lat=float(ser.lat),
            lon=float(ser.lon),
            min_lat=float(ser.min_lat),
            min_lon=float(ser.min_lon),
            max_lat=float(ser.max_lat),
            max_lon=float(ser.max_lon),
            velocity=float(ser.velocity),
        )
