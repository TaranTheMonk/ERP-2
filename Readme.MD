# ERP-2 Task Assignment in Spatial Crowdsourcing

## Synthetic Dataset
Raw data generate by: https://github.com/gmission/SCDataGenerator

Raw worker scheme:
```
id, lat, lon, capacity, activeness, [min_lat, min_lon, max_lat, max_lon], reliability, velocity
```
Raw task scheme:
```
lat, lon, arrival_time, expiry_time, requirement, confidence, entropy
```
worker scheme:
```
lat, lon, min_lat, min_lon, max_lat, max_lon, velocity
```
task scheme:
```
lat, lon, arrival_time, expiry_time, workload, expected_time, penalty_rate, reward
```

## Data Generation Rules
Task workload:
```
Uniformly from [2/5 * expiry_time, 2 * expiry_time]
```

Task expected_time:
```
Uniformly from [2/5 * expiry_time, 3/5 * expiry_time]
```

Task penalty_rate:
```
Uniformly from [0, reward / (expiry_time - expected_time)]
```

Task reward:
```
Uniformly from [0.0, 1.0]
```
