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

            //int test;
            //test = rand() % 10;
            //printf("%d", test);
            //printf(" \n");

            //int testTwo;
            //testTwo = rand() % 10;
            //printf("%d", testTwo);
            //printf(" \n");
        
            numBodies[i] = i;
            //printf("The following numBodies are: ");
            //printf("%d", numBodies[i]);
            //printf(" \n");

            bi[i].id = i;
            //printf("The following body ids are: ");
            //printf("%d", bi[i].id);
            //printf(" \n");            

            bi[i].mass = rand() % 10000000;
            printf("The following body masses are: %d\n", bi[i].mass);
            
            bi[i].radius = rand() % 10000000;
            //printf("The following body radii are: %d\n", bi[i].radius);

            bi[i].position.x = rand() % 3000;
            //printf("The following body x positions are: %d\n", bi[i].position.x);

            bi[i].position.y = rand() % 3000;
            //printf("The following body y positions are: %d\n", bi[i].position.y);

            bi[i].position.z = rand() % 3000;
            //printf("The following body z positions are: %d\n", bi[i].position.z);

            bi[i].velocity.x = rand() % 50000;
            //printf("The following body x velocities are: %d\n", bi[i].velocity.x);

            bi[i].velocity.y = rand() % 50000;
            //printf("The following body y velocities are: %d\n", bi[i].velocity.y);

            bi[i].velocity.z = rand() % 50000;
            //printf("The following body z velocities are: %d\n", bi[i].velocity.z);
        }
        
       // printf("yikers\n");
    }  
}

int timePrompt() {
    int timeChoice;
    printf("Enter 1 for one second/tick OR Enter 2 for two seconds/tick: ");
    scanf("%d", &timeChoice);
}

int main (int argc, char *argv[]) {
    int userChoice; 
    Timer timer; 

    struct body* bodies = (struct body*) malloc(sizeof(struct body*) * 2);

    bodies[0].id = 0;
    bodies[0].mass = 5.97e24; //earth mass
    bodies[0].radius = 6378e3;
    bodies[0].position.x = 0;
    bodies[0].position.y = 0;
    bodies[0].position.z = 0;
    bodies[0].velocity.x = 0;
    bodies[0].velocity.y = 0;
    bodies[0].velocity.z = 0;

    bodies[1].id = 1;
    bodies[1].mass = 300;
    bodies[1].radius = 10;
    bodies[1].position.x = 6378e3 + 37000e3; //GEO
    bodies[1].position.y = 0;
    bodies[1].position.z = 0;
    bodies[1].velocity.x = 0;
    bodies[1].velocity.y = 3e3; //GEO
    bodies[1].velocity.z = 0;
    
    //verify GPU implementation
    printf("CPU: ");
    print_float3(CPU_reduce_accel_vectors(bodies[0], bodies, 2));
    printf("\nGPU: ");
    print_float3(GPU_calculate_acceleration(bodies[0], bodies, 2));
    printf("\n"); 
   
    printf("Enter 1 for CPU calculations or 2 for GPU calculations: ");
    scanf("%d", &userChoice);
 
    if ("%d", userChoice == 1) {
        filePrompt();
        timePrompt();
        
        const int len = 2;
        
        printf("You chose to calculate using the CPU\n");
            
        unsigned long tick = 0;
        
        for(;;) {
            CPU_tick(bodies, len, 0.01);

            if (tick % 10000 == 0) {
                print_bodies(bodies, len, 4000e3);  
             
                printf("p:");
                print_float3(bodies[0].position);
                printf("v:");
                print_float3(bodies[0].velocity);
                printf(" p:");
                print_float3(bodies[1].position);
                printf("v:");
                print_float3(bodies[1].velocity);
                printf("\n");
            }
          
            tick++;
        }

    }
    if ("%d", userChoice == 2) {
        printf("You chose to calculate using the GPU\n");
        
        filePrompt();
        timePrompt();
        
        const int len = 2;
            
        unsigned long tick = 0;
        
        for(;;) {
            GPU_tick_improved(bodies, len, 1);

            if (tick % 100 == 0) {
                print_bodies(bodies, len, 4000e3);  
            
                printf("m:%e ", bodies[0].mass); 
                printf("p:");
                print_float3(bodies[0].position);
                printf("v:");
                print_float3(bodies[0].velocity);
                printf("m:%e ", bodies[1].mass); 
                printf(" p:");
                print_float3(bodies[1].position);
                printf("v:");
                print_float3(bodies[1].velocity);
                printf("\n");
            }
          
            tick++;
        }
    }    
    printf("haha lmao\n");
}
