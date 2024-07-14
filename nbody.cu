#include <math.h>
#include <stdio.h>
#include <stdlib.h>
// #include <omp.h>
// #include "timer.h" // Include the timer header
// #include "matric.h" // Include your custom matric.h header
#include <cuda_runtime.h>

#define SOFTENING 1e-9f

typedef struct {
    float x, y, z, vx, vy, vz;
} Body;

// Macro definitions
//#define THROUGHPUT(operations, seconds) ((operations) / (seconds) / 1e9) // GOPS
//#define RATIO_TO_PEAK_BANDWIDTH(actual_bandwidth, peak_bandwidth) ((actual_bandwidth) / (peak_bandwidth))

void randomizeBodies(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = 2.0f * ((float) rand() / RAND_MAX) - 1.0f;
    }
}

__global__
void bodyForceKernel(Body *p, float dt, int n, float *Fx, float *Fy, float *Fz) {
    //#pragma omp parallel for schedule(dynamic)

    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < n) {
        float Fx_reg = 0.0f;
        float Fy_reg = 0.0f;
        float Fz_reg = 0.0f;
        for (int j = 0; j < n; j++) {
            if (gid != j) {
                float dx = p[j].x - p[gid].x;
                float dy = p[j].y - p[gid].y;
                float dz = p[j].z - p[gid].z;
                float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                float invDist = rsqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;
                Fx_reg += dx * invDist3;
                Fy_reg += dy * invDist3;
                Fz_reg += dz * invDist3;
            }
        }
        Fx[gid] = Fx_reg;
        Fy[gid] = Fy_reg;
        Fz[gid] = Fz_reg;
        p[gid].vx += dt * Fx_reg;
        p[gid].vy += dt * Fy_reg;
        p[gid].vz += dt * Fz_reg;
    }
}

__global__
void pathIntegrationKernel(Body *p, float dt, int n) {
    int gid = threadIdx.x + blockIdx.x * blockDim.x;
    if (gid < n) {
        p[gid].x += p[gid].vx * dt;
        p[gid].y += p[gid].vy * dt;
        p[gid].z += p[gid].vz * dt;
    }
}

void saveForcesToFile(const char *filename, int nBodies, Body *p, float *Fx, float *Fy, float *Fz) {
    FILE *file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Unable to open file %s for writing.\n", filename);
        return;
    }
    for (int i = 0; i < nBodies; i++) {
        fprintf(file, "Body %d: x = %.3f, y = %.3f, z = %.3f, Fx = %.3f, Fy = %.3f, Fz = %.3f\n",
                i, p[i].x, p[i].y, p[i].z, Fx[i], Fy[i], Fz[i]);
    }
    fclose(file);
}

int main(int argc, char **argv) {
    int nBodies = 30000;
    if (argc > 2) nBodies = atoi(argv[2]);

    const float dt = 0.01f; // time step
    const int nIters = 10;  // simulation iterations

    int bytes = nBodies * sizeof(Body);
    Body *p = (Body *)malloc(bytes);

    if (p == NULL) {
        fprintf(stderr, "Unable to allocate memory for bodies.\n");
        return 1;
    }

    float *buf = (float *)malloc(6 * nBodies * sizeof(float));
    if (buf == NULL) {
        fprintf(stderr, "Unable to allocate memory for buffer.\n");
        free(p);
        return 1;
    }

    randomizeBodies(buf, 6 * nBodies);

    for (int i = 0; i < nBodies; i++) {
        p[i].x = buf[6 * i];
        p[i].y = buf[6 * i + 1];
        p[i].z = buf[6 * i + 2];
        p[i].vx = buf[6 * i + 3];
        p[i].vy = buf[6 * i + 4];
        p[i].vz = buf[6 * i + 5];
    }

    free(buf);

    Body *p_cuda;
    float *Fx_cuda, *Fy_cuda, *Fz_cuda;
    cudaMalloc(&p_cuda, nBodies * sizeof(Body));
    cudaMalloc(&Fx_cuda, nBodies * sizeof(float));
    cudaMalloc(&Fy_cuda, nBodies * sizeof(float));
    cudaMalloc(&Fz_cuda, nBodies * sizeof(float));

    cudaMemcpy(p_cuda, p, nBodies * sizeof(Body), cudaMemcpyHostToDevice);

    double totalTime = 0.0;

    int blockSize = 1024;
    if (argc > 1) blockSize = atoi(argv[1]);
    int numBlocks = (nBodies + blockSize - 1) / blockSize;

    for (int iter = 1; iter <= nIters; iter++) {
        cudaEvent_t start, stop;

        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        bodyForceKernel<<<numBlocks, blockSize>>>(p_cuda, dt, nBodies, Fx_cuda, Fy_cuda, Fz_cuda);
        pathIntegrationKernel<<<numBlocks, blockSize>>>(p_cuda, dt, nBodies);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop); // this also synchronizes the device

        float time_in_ms = 0;
        cudaEventElapsedTime(&time_in_ms, start, stop);
        if (iter > 1) {
            totalTime += time_in_ms / 1000.0;
        }
        printf("Iteration %d: %.3f seconds\n", iter, time_in_ms / 1000.0);
    }

    cudaMemcpy(p, p_cuda, nBodies * sizeof(Body), cudaMemcpyDeviceToHost);

    float *Fx = (float *)malloc(nBodies * sizeof(float));
    float *Fy = (float *)malloc(nBodies * sizeof(float));
    float *Fz = (float *)malloc(nBodies * sizeof(float));
    if (Fx == NULL || Fy == NULL || Fz == NULL) {
        fprintf(stderr, "Unable to allocate memory for force arrays.\n");
        free(p);
        if (Fx) free(Fx);
        if (Fy) free(Fy);
        if (Fz) free(Fz);
        return 1;
    }    
    cudaMemcpy(Fx, Fx_cuda, nBodies * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Fy, Fy_cuda, nBodies * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(Fz, Fz_cuda, nBodies * sizeof(float), cudaMemcpyDeviceToHost);

    saveForcesToFile("forces.txt", nBodies, p, Fx, Fy, Fz);

    double avgTime = totalTime / (double)(nIters - 1);
    double rate = (double)nBodies / avgTime;

    printf("Total time for iterations 2 through %d: %.3f seconds \n",
           nIters, totalTime);
    printf("Average rate for iterations 2 through %d: %.3f steps per second.\n",
           nIters, rate);
    printf("%d Bodies: average %0.3f Billion Interactions / second\n", nBodies, 1e-9 * nBodies * nBodies / avgTime);

    free(p);
    free(Fx);
    free(Fy);
    free(Fz);
    cudaFree(p_cuda);
    cudaFree(Fx_cuda);
    cudaFree(Fy_cuda);
    cudaFree(Fz_cuda);

    return 0;
}
