#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.h"
#include "body.h"

int filePrompt() {
    int fileChoice;
    int numBodies[256];
    struct body bi[256];
    srand(time(NULL));
    int r = rand() % 10;

    printf("Press 1 for solar system simulation or 2 for randomly generated simulation: ");
    scanf("%d", &fileChoice);

    if (fileChoice == 1) { 
        FILE* ptr; 
        char ch;
        ptr = fopen("bodydata.csv", "r");
    
        if (!ptr) {
            printf("File can't be opened\n");
        }

        printf("The files of this content contain:\n");

        while (!feof(ptr)) {
            ch = fgetc(ptr);
            printf("%c", ch);
        }
        fclose(ptr);
       // printf("yippee\n");
    }
    
    if (fileChoice == 2) {
        for (int i = 0; i < r; i++) {
            numBodies[i] = i;
            //printf("%d", numBodies[i]);
            bi[i].id = i;
            printf("%d", bi[i].id);   
        }
        
       // printf("yikers\n");
    }
      
}

int timePrompt() {
    int timeChoice;
    printf("Press 1 for the number of ticks or 2 seconds per tick: ");
    scanf("%d", &timeChoice);
}

int main (int argc, char *argv[]) {
    int userChoice; 
    Timer timer; 

    struct body* bodies[2];

    bodies[0] = (struct body*) malloc(sizeof(struct body));
    bodies[0]->id = 1;
    bodies[0]->mass = 5.97e24; //earth mass
    bodies[0]->radius = 6378e3;
    bodies[0]->position.x = 0;
    bodies[0]->position.y = 0;
    bodies[0]->position.z = 0;
    bodies[0]->velocity.x = 0;
    bodies[0]->velocity.y = 0;
    bodies[0]->velocity.z = 0;

    bodies[1] = (struct body*) malloc(sizeof(struct body));  
    bodies[1]->id = 2;
    bodies[1]->mass = 300;
    bodies[1]->radius = 10;
    bodies[1]->position.x = 6378e3 + 37000e3; //GEO
    bodies[1]->position.y = 0;
    bodies[1]->position.z = 0;
    bodies[1]->velocity.x = 0;
    bodies[1]->velocity.y = 3e3; //GEO
    bodies[1]->velocity.z = 0;
  
    //verify GPU implementation
    printf("CPU: ");
    print_float3(CPU_reduce_accel_vectors(bodies[0], bodies, 2));
    printf("\nGPU: ");
    print_float3(GPU_calculate_acceleration(bodies[0], bodies, 2));
    printf("\n"); 

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
        filePrompt();
        timePrompt();
        //simulator test code
        /*
        struct body* bodies[2];

        bodies[0] = &bodies[0];
        bodies[1] = &bodies[1];
        */
        const int len = 2;
        
        printf("You chose to calculate using the CPU\n");
            
        unsigned long tick = 0;
        
        for(;;) {
            CPU_tick(bodies, len, 0.01);

            if (tick % 100000 == 0) {
                print_bodies(bodies, len, 4000e3);  
             
                printf("p:");
                print_float3(bodies[0]->position);
                printf("v:");
                print_float3(bodies[0]->velocity);
                printf(" p:");
                print_float3(bodies[1]->position);
                printf("v:");
                print_float3(bodies[1]->velocity);
                printf("\n");
            }

            /* 
            printf("p:");
            print_float3(bodies[0]->position);
            printf("v:");
            print_float3(bodies[0]->velocity);
            printf(" p:");
            print_float3(bodies[1]->position);
            printf("v:");
            print_float3(bodies[1]->velocity);
            printf("\n");
            */
            
            tick++;
        }

    }
    if ("%d", userChoice == 2) {
        printf("You chose to calculate using the GPU\n");
        filePrompt();
        timePrompt();
    }    
 
    printf("haha lmao\n");
}
