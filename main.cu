#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.h"
#include "body.h"

int main (int argc, char *argv[]) {
    int userChoice;
   
    struct body b1;
    b1.id = 1;
    b1.mass = 5.97e24; //earth mass
    b1.position.x = 0;
    b1.position.y = 0;
    b1.position.z = 0;
    b1.velocity.x = 0;
    b1.velocity.y = 0;
    b1.velocity.z = 0;
    
    struct body b2;
    b2.id = 2;
    b2.mass = 300;
    b2.position.x = 6378e3 + 420e3; //LEO
    b2.position.y = 0;
    b2.position.z = 0;
    b2.velocity.x = 0;
    b2.velocity.y = 7.8e3; //LEO
    b2.velocity.z = 0;
   
    // Allocate and initlize host memory
    float *in_h, *out_h;
    float *in_d, *out_d;
    unsigned in_elements, out_elements;
    cudaError_t cuda_ret;
    dim3 dim_grid, dim_block;
    int i;

    out_elements = in_elements / (BLOCK_SIZE<<1);
    if(in_elements % (BLOCK_SIZE<<1)) out_elements++;
    
    out_h = (float*)malloc(out_elements * sizeof(float));
    if(out_h == NULL) FATAL("Unable to allocate host");

    printf("    Input size = %u\n", in_elements);

    // Allocate device variables
    cuda_ret = cudaMalloc((void**)&in_d, in_elements * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    cuda_ret = cudaMalloc((void**)&out_d, out_elements * sizeof(float));
    if(cuda_ret != cudaSuccess) FATAL("Unable to allocate device memory");
    printf("point %x, %x\n", in_d, out_d);

    cudaDeviceSynchronize();
 
    printf("Press 1 for CPU calculations or 2 for GPU calculations: ");
    scanf("%d", &userChoice);
    
    //printf("\nEntered number is: "%d", userChoice);
    
    if ("%d", userChoice == 1) {
        printf("You chose to calculate using the CPU\n");
    }
    if ("%d", userChoice == 2) {
        printf("You chose to calculate using the GPU\n");
    }    

    //simulator test code
    struct body* bodies[2];
    const int len = 2;
    
    bodies[0] = &b1;
    bodies[1] = &b2;

    for(;;) {
        CPU_tick(bodies, len, 0.01);
        printf("p:");
        print_float3(bodies[0]->position);
        printf("v:");
        print_float3(bodies[0]->velocity);
        printf(" p:");
        print_float3(bodies[1]->position);
        printf("v:");
        print_float3(bodies[1]->velocity);
        printf("\r");
    }
    printf("\n");
 
    printf("haha lmao\n");
}
