#include <stdio.h>
#include "body.h"

#define BLOCK_SIZE 16
__device__ void float3_atomicAdd(float3* f, float3 addend) {
    atomicAdd(&(f->x), addend.x);
    atomicAdd(&(f->y), addend.y);
    atomicAdd(&(f->z), addend.z);
}

//get distance between two bodies
__device__ float GPU_distance(struct body* b1, struct body* b2) {
    return sqrt(pow(b2->position.x - b1->position.x, 2) + 
                pow(b2->position.y - b1->position.y, 2) + 
                pow(b2->position.z - b1->position.z, 2));
}


//get gravity force magnitude between two bodies
__device__ float GPU_calculate_FG(struct body* b1, struct body* b2) {
    float G = 6.674e-11;
    float d = GPU_distance(b1, b2);
    float mag_F; 

    mag_F = (G * b1->mass * b2->mass)/pow(d, 2); //gravity formula

    return mag_F;
}

//get direction vector between two bodies
__device__ float3 GPU_get_direction_vector(struct body* origin, struct body* actor) {
    float3 direction;
    float norm = GPU_distance(origin, actor);

    direction = actor->position - origin->position;
    direction = direction / norm;

    return direction;
}

/* calculate acceleration of origin as exerted by actor */
__device__ float3 GPU_get_accel_vector(struct body* origin, struct body* actor) {
    float F = GPU_calculate_FG(origin, actor);
    float3 dir = GPU_get_direction_vector(origin, actor);

    float3 F_vec = dir * F; //get force vector
    float3 A_vec = F_vec / origin->mass; //F = MA -> A = F/M

    return A_vec;
}


//GPU kernel
__global__ void GPU_reduce_accel_vectors(float3* accel_out, struct body b, struct body* bodies, const unsigned int num_bodies) {
    float3 body_accel;

    unsigned int tx = threadIdx.x;
    unsigned int bx = blockIdx.x;

    unsigned int index = tx + (bx * BLOCK_SIZE);

    if (index < num_bodies) {
        if (b.id != bodies[index].id) {    
            body_accel = GPU_get_accel_vector(&b, &bodies[index]);
            float3_atomicAdd(accel_out, body_accel);
        }
    }

    __syncthreads();
}

float3 GPU_calculate_acceleration(struct body b, struct body* CPU_bodies, const unsigned int num_bodies) {
    cudaError_t cuda_ret;
    float3 CPU_accel;
    float3* GPU_accel;
    struct body* GPU_bodies;

    dim3 DimBlock(BLOCK_SIZE, 1, 1);
    dim3 DimGrid(ceil((float)num_bodies/((float)BLOCK_SIZE)), 1, 1);

    cudaMalloc((void**) &GPU_accel, sizeof(float3));
    cudaMalloc((void**) &GPU_bodies, sizeof(struct body) * num_bodies); //will be read-only
 
    cudaDeviceSynchronize();   
 
    CPU_accel.x = 0;
    CPU_accel.y = 0;
    CPU_accel.z = 0;
    
    cudaMemcpy(GPU_accel, &CPU_accel, sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(GPU_bodies, CPU_bodies, sizeof(struct body) * num_bodies, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();

    GPU_reduce_accel_vectors<<<DimGrid,DimBlock>>>(GPU_accel, b, GPU_bodies, num_bodies);

    cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cuda_ret));
        printf("Oppsie woopsie I did a fucky wucky (GPU kernel failed, lmao)\n");
    }

    cudaMemcpy(&CPU_accel, GPU_accel, sizeof(float3), cudaMemcpyDeviceToHost);
 
    cudaDeviceSynchronize();

    cudaFree(GPU_accel);
    cudaFree(GPU_bodies);
    
    cudaDeviceSynchronize();   

    return CPU_accel;
}

void GPU_tick(struct body* bodies, const int &num_bodies, const float &t) {
    float3 a;    

    for (int i = 0; i < num_bodies; i++) {
        a = GPU_calculate_acceleration(bodies[i], bodies, num_bodies);
        
        bodies[i].velocity = bodies[i].velocity + (a * (t/2.0)); //kick        
        bodies[i].position = bodies[i].position + (bodies[i].velocity * t); //drift
       
        a = GPU_calculate_acceleration(bodies[i], bodies, num_bodies);

        bodies[i].velocity = bodies[i].velocity + (a * (t/2.0)); //kick 
    }
}

