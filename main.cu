#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.h"
#include "body.h"

#define DISTANCE_SCALE 30000000
#define MAX_LINE_LENGTH 100

struct body bi[256];
int randomizedChoice = 0;
int numBodies;

int filePrompt() {
    int fileChoice;
    char fileName[100];

    printf("Press 1 for simulation from file or 2 for randomly generated simulation: ");
    scanf("%d", &fileChoice);

    if (fileChoice == 1) {
       printf("Type in the file name you would like to simulate: ");
       scanf("%s", &fileName);  
       FILE *file = fopen(fileName, "r");
       if (file == NULL) {
            printf("Failed to open file\n");
            return 1; 
        }

        char line[MAX_LINE_LENGTH];
        fgets(line, MAX_LINE_LENGTH, file);
        fgets(line, MAX_LINE_LENGTH, file);

        int i = 0;
        float temp_mass;
        float temp_radius;
        float3 temp_position;
        float3 temp_velocity;

        while(fgets(line, MAX_LINE_LENGTH, file) != NULL && i < 10) {
            printf(line);
            printf("\n");
            
            sscanf(line, "%f,%f,%f,%f,%f,%f,%f,%f", 
                &temp_mass,
                &temp_radius,
                &temp_position.x,
                &temp_position.y,
                &temp_position.z,
                &temp_velocity.x,
                &temp_velocity.y,
                &temp_velocity.z);
            
            bi[i].id = i;
            bi[i].mass = (float) temp_mass;
            bi[i].radius = (float) temp_radius;
            
            bi[i].position.x = (float) temp_position.x;
            bi[i].position.y = (float) temp_position.y;
            bi[i].position.z = (float) temp_position.z;

            bi[i].velocity.x = (float) temp_velocity.x;
            bi[i].velocity.y = (float) temp_velocity.y;
            bi[i].velocity.z = (float) temp_velocity.z;

            i++;
        }
        fclose(file);

        numBodies = i;
              
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
            bi[i].id = i;
            bi[i].mass = (rand() % 1000) * (10e20);
           
            bi[i].radius = rand() % 1000;
            bi[i].position.x = (rand() % DISTANCE_SCALE) - (DISTANCE_SCALE / 2);
            bi[i].position.y = (rand() % DISTANCE_SCALE) - (DISTANCE_SCALE / 2);
            bi[i].position.z = (rand() % DISTANCE_SCALE) - (DISTANCE_SCALE / 2);
            bi[i].velocity.x = rand() % 10000 - (10000 / 2);
            bi[i].velocity.y = rand() % 10000 - (10000 / 2);
            bi[i].velocity.z = rand() % 10000 - (10000 / 2);
        }
    } 

    /*
    for (int j = 0; j < numBodies; j++) {
        printf("Body %d:\n", bi[j].id);
        printf("Mass %e\n", bi[j].mass);
        printf("Radius: %e\n", bi[j].radius);
        printf("X Position: %e\n", bi[j].position.x);
        printf("Y Position: %e\n", bi[j].position.y);
        printf("Z Position %e\n", bi[j].position.z);
        printf("X Velocity %e\n", bi[j].velocity.x);
        printf("Y Velocity %e\n", bi[j].velocity.y);
        printf("Z Velocity %e\n", bi[j].velocity.z);
    }
    */    
}

int main (int argc, char *argv[]) {
    int userChoice; 
    Timer timer; 

    printf("Press 1 for CPU calculations or 2 for GPU calculations: ");
    scanf("%d", &userChoice);
    filePrompt();
 
    int len = numBodies;

    unsigned long long tick = 0;

    unsigned int ticks_per_display = 1000;

    unsigned long long max_ticks = timerPrompt(); 

    float secs_per_tick = tickTime(); //0.1 by default

    /* auto scaling code */
    struct body origin;
    origin.position.x = 0;
    origin.position.y = 0;
    origin.position.z = 0;
    float max_distance = 0;
    float autoscale;

    for (int i = 0; i < numBodies; i++) {
        if (distance(&origin, &bi[i]) > max_distance) {
            max_distance = distance(&origin, &bi[i]);   
        }
    }

    autoscale = (max_distance * 2.0) / 40.0;

    startTime(&timer);
    /* main while loop */
    while (tick < max_ticks) {
        if ("%d", userChoice == 1) {
            CPU_tick(bi, len, secs_per_tick);
            len = CPU_collisions(bi, len);
        } else if ("%d", userChoice == 2) {
            GPU_tick_improved(bi, len, secs_per_tick); 
        }    

        if (tick % ticks_per_display == 0) {
            /*
            //scale bodies
            max_distance = 0; 
            for (int i = 0; i < numBodies; i++) {
                if (distance(&origin, &bi[i]) > max_distance) {
                    max_distance = distance(&origin, &bi[i]);   
                }
            }

            autoscale = (max_distance * 2.0) / 40.0;
            */

            print_bodies(bi, len, autoscale);//DISTANCE_SCALE/40);
            //print_bodies_numbered(bi, len, DISTANCE_SCALE/40);
            printf("Bodies:%d, Scale=%e meters, Tick=%lu\n", len, autoscale, tick);
            print_body(&bi[0]);
        }

        tick++;
    }
    stopTime(&timer);
    printf("%f s\n",elapsedTime(timer));
    
    printf("haha lmao\n");
}
