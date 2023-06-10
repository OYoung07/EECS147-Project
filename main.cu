#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.h"
#include "body.h"

#define DISTANCE_SCALE 30000000

struct body bi[256];
struct body solarSystem[10];
int randomizedChoice = 0;
int numBodies;

int filePrompt() {
    int fileChoice;

    printf("Press 1 for solar system simulation or 2 for randomly generated simulation: ");
    scanf("%d", &fileChoice);

    if (fileChoice == 1) {  
        char symbol;
        unsigned char symbol2;  
        FILE *FileIn;
        FileIn = fopen("bodydata.csv", "rt");
        while ((symbol=getc(FileIn))!=EOF) {
            symbol2 = (unsigned char) symbol;
            if (symbol2 >= '0' && symbol2 <= '9') {
                printf("%c", symbol);
            }
        }
       // printf("yippee\n");
    }
    
    if (fileChoice == 2) {
        int seedNum;
        randomizedChoice = 1;

        printf("Enter a seed number: ");
        scanf("%d", &seedNum);
        srand(seedNum);
        printf("Enter the number of bodies you want to simulate: ");
        scanf("%d", &numBodies);
    
        for (int i = 0; i < numBodies; i++) {

            //int test;
            //test = rand() % 10;
            //printf("%d", test);
            //printf(" \n");

            //int testTwo;
            //testTwo = rand() % 10;
            //printf("%d", testTwo);
            //printf(" \n");

            bi[i].id = i;
            //printf("The following body ids are: ");
            //printf("%d", bi[i].id);
            //printf(" \n");            

            bi[i].mass = (rand() % 1000) * (10e20);
            //printf("The following body masses are: %d\n", bi[i].mass);
            
            bi[i].radius = rand() % 1000;
            //printf("The following body radii are: %d\n", bi[i].radius);

            bi[i].position.x = (rand() % DISTANCE_SCALE) - (DISTANCE_SCALE / 2);
            //printf("The following body x positions are: %d\n", bi[i].position.x);

            bi[i].position.y = (rand() % DISTANCE_SCALE) - (DISTANCE_SCALE / 2);
            //printf("The following body y positions are: %d\n", bi[i].position.y);

            bi[i].position.z = (rand() % DISTANCE_SCALE) - (DISTANCE_SCALE / 2);
            //printf("The following body z positions are: %d\n", bi[i].position.z);

            bi[i].velocity.x = rand() % 10000 - (10000 / 2);
            //printf("The following body x velocities are: %d\n", bi[i].velocity.x);

            bi[i].velocity.y = rand() % 10000 - (10000 / 2);
            //printf("The following body y velocities are: %d\n", bi[i].velocity.y);

            bi[i].velocity.z = rand() % 10000 - (10000 / 2);
            //printf("The following body z velocities are: %d\n", bi[i].velocity.z);
        }
        
       // printf("yikers\n");
    }  
}

int main (int argc, char *argv[]) {
    int userChoice; 
    Timer timer; 

    printf("Press 1 for CPU calculations or 2 for GPU calculations: ");
    scanf("%d", &userChoice);
    filePrompt();
 
    const int len = numBodies;

    unsigned long long tick = 0;
    float secs_per_tick = 0.1;
    unsigned int ticks_per_display = 1000;

    unsigned long long max_ticks = timerPrompt(); 

    /* main while loop */
    while (tick < max_ticks) {
        if ("%d", userChoice == 1) {
            CPU_tick(bi, len, secs_per_tick);
        } else if ("%d", userChoice == 2) {
            GPU_tick_improved(bi, len, secs_per_tick); 
        }    

        if (tick % ticks_per_display == 0) {
            print_bodies(bi, len, DISTANCE_SCALE/40);
        }

        tick++;
    }

    printf("haha lmao\n");
}
