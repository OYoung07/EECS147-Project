#include <stdio.h>
#include "body.h"

#define BLOCK_SIZE 256
#define SHARED_MEM_SIZE 256

//get distance between two bodies
__device__ double GPU_distance(struct body* b1, struct body* b2) {
    return sqrt(pow(b2->position.x - b1->position.x, 2) + 
                pow(b2->position.y - b1->position.y, 2) + 
                pow(b2->position.z - b1->position.z, 2));
}


//get gravity force magnitude between two bodies
__device__ double GPU_calculate_FG(struct body* b1, struct body* b2) {
    double G = 6.674e-11;
    double d = GPU_distance(b1, b2);
    double mag_F; 

    mag_F = (G *(double)b1->mass *(double)b2->mass)/pow(d, 2); //gravity formula

    return (double)mag_F;
}

//get direction vector between two bodies
__device__ double3 GPU_get_direction_vector(struct body* origin, struct body* actor) {
    double3 direction;
    double norm = GPU_distance(origin, actor);

    direction = actor->position - origin->position;
    direction = direction / norm;

    return direction;
}

/* calculate acceleration of origin as exerted by actor */
__device__ double3 GPU_get_accel_vector(struct body* origin, struct body* actor) {
    double F = GPU_calculate_FG(origin, actor);
    double3 dir = GPU_get_direction_vector(origin, actor);

    double3 F_vec = dir * F; //get force vector
    double3 A_vec = F_vec / origin->mass; //F = MA -> A = F/M

    return A_vec;
}

//better GPU kernel
__global__ void GPU_tick_shared_memory(struct body* output_bodies, const unsigned int num_bodies, const double t, unsigned int* collisions) {
    __shared__ struct body temp_bodies_shared[SHARED_MEM_SIZE]; 

    unsigned int tx = threadIdx.x;
    unsigned int bx = blockIdx.x;

    unsigned int index = tx + (bx * BLOCK_SIZE);

    double3 a;

    if (index < num_bodies) { //populate shared memory
        temp_bodies_shared[index] = output_bodies[index]; 
    }

    __syncthreads();

    //do calculations 
    if (index < num_bodies) {
        //get first acceleration
        a.x = 0; a.y = 0; a.z = 0;
        for (int i = 0; i < num_bodies; i++) { 
            if (output_bodies[index].id != temp_bodies_shared[i].id) {    
                a = a + GPU_get_accel_vector(&output_bodies[index], &temp_bodies_shared[i]);
            }
        }
        
        output_bodies[index].velocity = output_bodies[index].velocity + (a * (t/2.0)); //kick        
        output_bodies[index].position = output_bodies[index].position + (output_bodies[index].velocity * t); //drift
                  
        //get second acceleration
        a.x = 0; a.y = 0; a.z = 0;
        for (int i = 0; i < num_bodies; i++) { 
            if (output_bodies[index].id != temp_bodies_shared[i].id) {    
                a = a + GPU_get_accel_vector(&output_bodies[index], &temp_bodies_shared[i]);
            }
        }
        
        output_bodies[index].velocity = output_bodies[index].velocity + (a * (t/2.0)); //kick 
    }

    __syncthreads();

    //do collisions here
    if (index < num_bodies) {
        for (int i = 0; i < num_bodies; i++) {
            if ((GPU_distance(&output_bodies[index], &output_bodies[i]) < (output_bodies[index].radius + output_bodies[i].radius)) && (index != i)) {
                atomicAdd(collisions, 1); //add to collisions       
            }
        }
    }
}

unsigned int GPU_tick_improved(struct body* CPU_bodies, unsigned int num_bodies, const double &t) {
    cudaError_t cuda_ret;
    struct body* GPU_bodies;

    unsigned int* GPU_collisions;
    unsigned int collisions;
    
    struct body new_bodies[128];
    unsigned int new_bodies_index = 0;

    collisions = 0;

    /* GPU inits and kernel execution */
    cudaMalloc((void**) &GPU_collisions, sizeof(unsigned int));
    cudaMalloc((void**) &GPU_bodies, sizeof(struct body) * num_bodies); 
    cudaDeviceSynchronize();   
 
    cudaMemcpy(GPU_collisions, &collisions, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(GPU_bodies, CPU_bodies, sizeof(struct body) * num_bodies, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
 
    dim3 DimBlock(BLOCK_SIZE, 1, 1);
    dim3 DimGrid(ceil((double)num_bodies/((double)BLOCK_SIZE)), 1, 1);
    GPU_tick_shared_memory<<<DimGrid,DimBlock>>>(GPU_bodies, num_bodies, t, GPU_collisions);

    cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(cuda_ret));
        printf("Oppsie woopsie I did a fucky wucky (GPU kernel failed, lmao)\n");
    }

    cudaMemcpy(&collisions, GPU_collisions, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(CPU_bodies, GPU_bodies, sizeof(struct body) * num_bodies, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaFree(GPU_collisions);
    cudaFree(GPU_bodies);

    if (collisions > 0) {
       num_bodies = CPU_collisions(CPU_bodies, num_bodies); 
    }

    return num_bodies;
}
