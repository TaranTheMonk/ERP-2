# ERP-2 Task Assignment in Spatial Crowdsourcing

## Synthetic Dataset
Raw data generate by: https://github.com/gmission/SCDataGenerator

Raw worker data template:
```
id; lat; lon; capacity; activeness; [min_lat, min_lon, max_lat, max_lon]; reliability; velocity
```
Raw task data template:
```
lat; lon; arrival_time; expiry_time; requirement; confidence; entropy
```
worker scheme:
```
lat, lon, min_lat, min_lon, max_lat, max_lon, velocity
```
task scheme:
```
lat, lon, deadline, workload, expected_time, penalty_rate, reward
```

## Data Generation Rules
Task deadline:
```
expiry_time - arrival_time
```

Task workload:
```
Uniformly from [2/5 * deadline, 2 * deadline]
```

Task expected_time:
```
Uniformly from [2/5 * deadline, 3/5 * deadline]
```

Task penalty_rate:
```
Uniformly from [0, reward / (deadline - expected_time)]
```

Task reward:
```
Uniformly from [0.0, 1.0]
```
