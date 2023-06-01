#include <stdio.h>
#include "body.h"

#define BLOCK_SIZE 16

//GPU kernel
//TODO: need to redefine body functions as host functions
/*
__global__ void GPU_reduce_accel_vectors(float3* accel_out, struct body* b, struct body** bodies, const unsigned  int num_bodies) {
    float3 accel;
    accel.x = 0;
    accel.y = 0;
    accel.z = 0;    
 
    __shared__ float3 partialSum[2 * BLOCK_SIZE];

    unsigned int tx = threadIdx.x;
    unsigned int start = 2 * blockIdx.x * blockDim.x;

    if (bodies[tx]->id == b->id) {
        return; //exit if current body is self
    }

    partialSum[tx] = get_accel_vector(b, bodies[start + tx]);
    partialSum[blockDim.x + tx] = get_accel_vector(b, bodies[start + blockDim.x + tx]);

    for (unsigned int stride = blockDim.x; stride > 0; stride /= 2) {
        __syncthreads();

        if (tx < stride) {
            partialSum[tx] = partialSum[tx] +  partialSum[tx + stride];
        }
    }

    if (tx == 0) {
        atomicAdd(&accel_out->x, partialSum[0].x); //add back to output without race condition
        atomicAdd(&accel_out->y, partialSum[0].y);
        atomicAdd(&accel_out->z, partialSum[0].z);
    }
}

float3 GPU_calculate_acceleration(struct body** CPU_bodies, struct body* CPU_b, const unsigned int num_bodies) {
    float3 CPU_accel;
    float3* GPU_accel;
    struct body* GPU_b;
    struct body** GPU_bodies;

    dim3 DimBlock(BLOCK_SIZE);
    dim3 DimGrid(num_bodies/(BLOCK_SIZE * 2));

    cudaMalloc((void**) &GPU_accel, sizeof(float3));
    cudaMalloc((void**) &GPU_b, sizeof(struct body)); //will be written to
    cudaMalloc((void**) &GPU_bodies, sizeof(struct body) * num_bodies); //will be read-only
    
    CPU_accel.x = 0;
    CPU_accel.y = 0;
    CPU_accel.z = 0;
    
    cudaMemcpy(&GPU_accel, &CPU_accel, sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(GPU_b, CPU_b, sizeof(struct body), cudaMemcpyHostToDevice);
    cudaMemcpy(GPU_bodies, CPU_bodies, sizeof(struct body) * num_bodies, cudaMemcpyHostToDevice);

    GPU_reduce_accel_vectors<<<DimGrid,DimBlock>>>(GPU_accel, GPU_b, GPU_bodies, num_bodies);

    cudaMemcpy(&CPU_accel, GPU_accel, sizeof(float3), cudaMemcpyDeviceToHost);
    cudaFree(GPU_accel);
    cudaFree(GPU_b);
    cudaFree(GPU_bodies);
    
    return CPU_accel;
}
*/

//need to allocate GPU memory for bodies and accel_out
/*
void GPU_tick(struct body* CPU_bodies, const int &num_bodies) {
    float3 CPU_a; //acceleration scalar
    float3 GPU_a;
    struct body* GPU_bodies; //deivce memory for computation
    struct body* GPU_body_outputs; //array to write to

    cudaMalloc((void**) &GPU_bodies, sizeof(struct body) * num_bodies); //read-only
    cudaMalloc((void**) &GPU_body_outputs, sizeof(struct body) * num_bodies); //write to

    cudaMemcpy(GPU_bodies, CPU_bodies, sizeof(struct body) * num_bodies, cudaMemcpyHostToDevice);
    cudaMemcpy(GPU_body_outputs, CPU_bodies, sizeof(struct body) * num_bodiesi, cudaMemcpyHostToDevice);

    for (int i = 0; i < num_bodies, i++) {
        //copy to GPU and run acceleration calculations
        CPU_a = 0;
        cudaMemcpy(GPU_a, CPU_a, sizeof(float3), cudaMemcpyHostToDevice);    

        GPU_reduce_accel_vectors(GPU_a, GPU_body_outputs[i], GPU_bodies, num_bodies);

        cudaMemcpy(CPU_a, GPU_a, sizeof(float3), cudaMemcpyHostToDevice);
 
        GPU_bo       
    }

    //just for porting, delete this late:
    for (int i = 0; i < num_bodies; i++) {
        a = CPU_reduce_accel_vectors(bodies[i], bodies, num_bodies);
        
        bodies[i]->velocity = bodies[i]->velocity + (a * (t/2.0)); //kick        
        bodies[i]->position = bodies[i]->position + (bodies[i]->velocity * t); //drift
       
        a = CPU_reduce_accel_vectors(bodies[i], bodies, num_bodies);

        bodies[i]->velocity = bodies[i]->velocity + (a * (t/2.0)); //kick 
    }
    cudaMemcpy(GPU_bodies, bodies, sizeof(struct body) * num_bodies);
}
*/
