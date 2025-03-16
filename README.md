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

## Performance
The program reports execution time and accuracy compared to the true value of Pi.
