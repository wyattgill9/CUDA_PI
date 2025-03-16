#include "hip/hip_runtime.h"
#include "hip/hip_runtime.h"
/***********************************************
 * CUDA Monte Carlo Pi Estimation
 *
 * This project demonstrates how to use CUDA for
 * a Monte Carlo calculation to estimate Pi.
 * FOR PI DAY 2025 - Wyatt Gill
 ***********************************************/

#include <stdio.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>
#include <hiprand/hiprand_kernel.h>
#include <time.h>

#define CUDA_CHECK(call) \
do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "CUDA error in %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void estimate_pi(unsigned int seed, unsigned int iterations, unsigned int *count) {
    // Set up CUDA random number generator
    hiprandState state;

    // Each thread gets its own random number generator with a different seed
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    hiprand_init(seed + idx, 0, 0, &state);

    unsigned int local_count = 0;

    for (int i = 0; i < iterations; i++) {
        // Generate random points in the unit square [0,1] x [0,1]
        float x = hiprand_uniform(&state);
        float y = hiprand_uniform(&state);

        // Check if the point is inside the unit circle
        if (x*x + y*y <= 1.0f) {
            local_count++;
        }
    }

    atomicAdd(count, local_count);
}

int main(int argc, char *argv[]) {
    unsigned int num_blocks = 256;
    unsigned int num_threads = 256;
    unsigned int iterations_per_thread = 10000;
    // Parse command line arguments if provided                                                 if (argc > 1) num_blocks = atoi(argv[1]);
    if (argc > 2) num_threads = atoi(argv[2]);                                                  if (argc > 3) iterations_per_thread = atoi(argv[3]);

    // Calc total iterations
    unsigned long long total_iterations = (unsigned long long)num_blocks * num_threads * iterations_per_thread;

    printf("CUDA Monte Carlo Pi Estimation\n");
    printf("------------------------------\n");
    printf("Blocks: %u\n", num_blocks);
    printf("Threads per block: %u\n", num_threads);
    printf("Iterations per thread: %u\n", iterations_per_thread);
    printf("Total iterations: %llu\n", total_iterations);
    printf("------------------------------\n");

    // Alloc device memory
    unsigned int *d_count;
    CUDA_CHECK(hipMalloc(&d_count, sizeof(unsigned int)));
    CUDA_CHECK(hipMemset(d_count, 0, sizeof(unsigned int)));

    clock_t start = clock();

    // Launch kernel
    unsigned int seed = time(NULL);
    estimate_pi<<<num_blocks, num_threads>>>(seed, iterations_per_thread, d_count);

    // Check for errors
    CUDA_CHECK(hipGetLastError());

    // Wait for kernel to finish
    CUDA_CHECK(hipDeviceSynchronize());

    // Cpy res back to host
    unsigned int h_count = 0;
    CUDA_CHECK(hipMemcpy(&h_count, d_count, sizeof(unsigned int), hipMemcpyDeviceToHost));

    // calc Pi
    double pi_estimate = 4.0 * h_count / total_iterations;

    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    // Results
    printf("Points inside circle: %u\n", h_count);
    printf("Pi estimate: %.10f\n", pi_estimate);
    printf("True Pi: 3.1415926536\n");
    printf("Error: %.10f\n", fabs(pi_estimate - 3.1415926536));
    printf("Execution time: %.2f seconds\n", elapsed);
    printf("Performance: %.2f million iterations per second\n", total_iterations / elapsed / 1000000.0);

    CUDA_CHECK(hipFree(d_count));

    return 0;
}
