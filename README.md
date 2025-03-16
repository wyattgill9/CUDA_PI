# CUDA Monte Carlo Pi Estimation PI DAY 2025

A CUDA implementation of Monte Carlo Pi estimation that GPU parallelism.

## Requirements
- NVIDIA GPU with CUDA support || AMD GPU with ROCm support + Hipify
- CUDA Toolkit (8.0+)

## Usage

Build:
```bash
whatever way you like to build your project ie hipify for AMD or nvcc for NVIDIA
```

Run:
```bash
./monte_carlo_pi [blocks] [threads] [iterations_per_thread]
```

- Randomly generates points in a square
- Counts points falling inside a unit circle
- Estimates Pi as: 4 × (points inside circle) / (total points)

## Performance (7900XT & 9 9900x)
```bash
CUDA Monte Carlo Pi Estimation
------------------------------
Blocks: 256
Threads per block: 256
Iterations per thread: 10000
Total iterations: 655360000
------------------------------
Points inside circle: 514718848
Pi estimate: 3.1415945312
True Pi: 3.1415926536
Error: 0.0000018776
Execution time: 0.00 seconds
Performance: 136504.89 million iterations per second
```
