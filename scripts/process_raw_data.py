import os
import random
import pandas
import pathlib

RESOURCE_PATH = os.path.join(pathlib.Path(__file__).parent, "../.resources")


def process_worker_data(size: int):
    for raw_worker_file in os.listdir(
        os.path.join(RESOURCE_PATH, f"raw_data/worker_{size}")
    ):
        # read raw data
        with open(
            os.path.join(RESOURCE_PATH, f"raw_data/worker_{size}", raw_worker_file), "r"
        ) as f:
            lines = f.readlines()

        # process raw data
        df = {
            "lat": [],
            "lon": [],
            "min_lat": [],
            "min_lon": [],
            "max_lat": [],
            "max_lon": [],
            "velocity": [],
        }
        for line in lines:
            # data template
            # id; lat; lon; capacity; activeness; [min_lat, min_lon, max_lat, max_lon]; reliability; velocity
            line = line.split(";")
            df["lat"].append(float(line[1]))
            df["lon"].append(float(line[2]))
            df["min_lat"].append(float(line[5][1:]))
            df["min_lon"].append(float(line[6]))
            df["max_lat"].append(float(line[7]))
            df["max_lon"].append(float(line[8][:-1]))
            df["velocity"].append(float(line[10]))

        # write processed data
        df = pandas.DataFrame(df)
        df.to_csv(
            os.path.join(
                RESOURCE_PATH,
                f"processed_data/worker_{size}",
                raw_worker_file.split(".")[0] + ".csv",
            ),
            index=False,
        )


def process_task_data(size: int):
    for raw_worker_file in os.listdir(
        os.path.join(RESOURCE_PATH, f"raw_data/task_{size}")
    ):
        # read raw data
        with open(
            os.path.join(RESOURCE_PATH, f"raw_data/task_{size}", raw_worker_file), "r"
        ) as f:
            lines = f.readlines()

        # process raw data
        df = {
            "lat": [],
            "lon": [],
            "deadline": [],
            "workload": [],
            "expected_time": [],
            "penalty_rate": [],
            "reward": [],
        }
        for line in lines:
            # data template
            # lat; lon; arrival_time; expiry_time; requirement; confidence; entropy
            line = line.split(";")
            deadline = float(line[3]) - float(line[2])
            expected_time = random.uniform(2 / 5 * deadline, 3 / 5 * deadline)
            reward = random.uniform(0.0, 1.0)

            df["lat"].append(float(line[0]))
            df["lon"].append(float(line[1]))
            df["deadline"].append(deadline)
            df["workload"].append(random.uniform(2 / 5 * deadline, 2 * deadline))
            df["expected_time"].append(expected_time)
            df["penalty_rate"].append(
                random.uniform(0.0, reward / (deadline - expected_time))
            )
            df["reward"].append(reward)

        # write processed data
        df = pandas.DataFrame(df)
        df.to_csv(
            os.path.join(
                RESOURCE_PATH,
                f"processed_data/task_{size}",
                raw_worker_file.split(".")[0] + ".csv",
            ),
            index=False,
        )


if __name__ == "__main__":
    process_worker_data(size=200)
    process_task_data(size=200)
