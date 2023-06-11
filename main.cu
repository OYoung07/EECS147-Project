#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.h"
#include "body.h"

#define DISTANCE_SCALE 30000000
#define MAX_LINE_LENGTH 100

struct body bi[256];
struct body solarSystem[10];
int randomizedChoice = 0;
int numBodies;

int filePrompt() {
    int fileChoice;

    printf("Press 1 for simulation from file or 2 for randomly generated simulation: ");
    scanf("%d", &fileChoice);

    if (fileChoice == 1) {  
       FILE *file = fopen("bodydata.csv", "r");
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
            
            solarSystem[i].id = i;
            solarSystem[i].mass = temp_mass;
            solarSystem[i].radius = temp_radius;
            solarSystem[i].position = temp_position;
            solarSystem[i].velocity = temp_velocity;

            i++;
        }
        fclose(file);

        for (int j = 0; j < i; j++) {
            printf("Body %d:\n", solarSystem[j].id);
            printf("Mass %e\n", solarSystem[j].mass);
            printf("Radius: %e\n", solarSystem[j].radius);
            printf("X Position: %e\n", solarSystem[j].position.x);
            printf("Y Position: %e\n", solarSystem[j].position.y);
            printf("Z Position %e\n", solarSystem[j].position.z);
            printf("X Velocity %e\n", solarSystem[j].velocity.x);
            printf("Y Velocity %e\n", solarSystem[j].velocity.y);
            printf("Z Velocity %e\n", solarSystem[j].velocity.z);
        }
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
            CPU_collisions(bi, len);
        } else if ("%d", userChoice == 2) {
            GPU_tick_improved(bi, len, secs_per_tick); 
        }    

        if (tick % ticks_per_display == 0) {
            print_bodies(bi, len, DISTANCE_SCALE/40);
            //print_bodies_numbered(bi, len, DISTANCE_SCALE/40);
        }

        tick++;
    }

    printf("haha lmao\n");
}
