#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.h"
#include "body.h"

int main (int argc, char *argv[]) {
    int userChoice;
   
    struct body b1;
    b1.mass = 100;
    b1.position.x = 0;
    b1.position.y = 0;
    b1.position.z = 0;
    
    struct body b2;
    b2.mass = 200;
    b2.position.x = 100;
    b2.position.y = 10000;
    b2.position.z = 100;
    
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

    printf("%.15f\n",calculate_FG(&b1,&b2));
   
    float3 vec = get_direction_vector(&b1,&b2);

    printf("(%.5f,%.5f,%.5f)\n",vec.x,vec.y,vec.z);

    //float3 accel = get_accel_vector(&b1, &b2);

    //printf("(%.5f,%.5f,%.5f)\n",vec.x,vec.y,vec.z);
}
