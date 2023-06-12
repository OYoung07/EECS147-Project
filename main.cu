#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.h"
#include "body.h"

#define DISTANCE_SCALE 30000000
#define MAX_LINE_LENGTH 100

struct body bi[256];
int numBodies;
int seedNum;
FILE *fp = fopen("outputData.csv", "w");
double data[6];

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
        double temp_mass;
        double temp_radius;
        double3 temp_position;
        double3 temp_velocity;

        while(fgets(line, MAX_LINE_LENGTH, file) != NULL && i < 10) {
            printf(line);
            printf("\n");
            
            sscanf(line, "%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf", 
                &temp_mass,
                &temp_radius,
                &temp_position.x,
                &temp_position.y,
                &temp_position.z,
                &temp_velocity.x,
                &temp_velocity.y,
                &temp_velocity.z);
            
            bi[i].id = i;
            bi[i].mass = (double) temp_mass;
            bi[i].radius = (double) temp_radius;
            
            bi[i].position.x = (double) temp_position.x;
            bi[i].position.y = (double) temp_position.y;
            bi[i].position.z = (double) temp_position.z;

            bi[i].velocity.x = (double) temp_velocity.x;
            bi[i].velocity.y = (double) temp_velocity.y;
            bi[i].velocity.z = (double) temp_velocity.z;

            i++;
        }
        fclose(file);

        numBodies = i;
              
    }
    
    if (fileChoice == 2) {
        printf("Enter a seed number: ");
        scanf("%d", &seedNum);
        srand(seedNum);
        printf("Enter the number of bodies you want to simulate [MAX:256] : ");
        scanf("%d", &numBodies);

        /* 
        bi[0].mass = 1e24;
        bi[0].radius = 2000e3;
        bi[0].position.x = 0;
        bi[0].position.y = 0;
        bi[0].position.z = 0;
        bi[0].velocity.x = 0;//rand() % 10000 - (10000 / 2);
        bi[0].velocity.y = 0;//rand() % 10000 - (10000 / 2);
        bi[0].velocity.z = 0;//rand() % 10000 - (10000 / 2);
        */

        for (int i = 0; i < numBodies; i++) {
            bi[i].id = i;
            bi[i].mass = (rand() % 1000) * (10e20);
           
            bi[i].radius = (rand() % 100000);
            bi[i].position.x = (rand() % DISTANCE_SCALE) - (DISTANCE_SCALE / 2);
            bi[i].position.y = (rand() % DISTANCE_SCALE) - (DISTANCE_SCALE / 2);
            bi[i].position.z = (rand() % DISTANCE_SCALE) - (DISTANCE_SCALE / 2);
            bi[i].velocity.x = 0;//rand() % 10000 - (10000 / 2);
            bi[i].velocity.y = 0;//rand() % 10000 - (10000 / 2);
            bi[i].velocity.z = 0;//rand() % 10000 - (10000 / 2);
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

void collisionPrompt(int *do_collisions) {
    printf("Press 1 for inelastic collisions and 0 for no collisions: ");
    scanf("%d", do_collisions);
    //printf("%d", do_collisions);
}

int main (int argc, char *argv[]) {
    int userChoice; 
    Timer timer; 

    printf("Press 1 for CPU calculations or 2 for GPU calculations: ");
    scanf("%d", &userChoice);
    filePrompt();
 
    int do_collisions = 1;

    int len = numBodies;

    unsigned long long tick = 0;

    unsigned int ticks_per_display = ticksPerDisplay();
    
    unsigned long long max_ticks = timerPrompt(); 
    
    collisionPrompt(&do_collisions);

    double secs_per_tick = tickTime(); //1 by default

    /* auto scaling code */
    struct body origin;
    origin.position.x = 0;
    origin.position.y = 0;
    origin.position.z = 0;
    double max_distance = 0;
    double autoscale;

    double totalmass;
    double totalenergy_k;
    double totalenergy_p;

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
            if (do_collisions == 1) {
                len = CPU_collisions(bi, len);
            }
        } else if ("%d", userChoice == 2) {
            len = GPU_tick_improved(bi, len, secs_per_tick, do_collisions); 
        }    

        if (tick % ticks_per_display == 0) {
            print_bodies(bi, len, autoscale);//DISTANCE_SCALE/40);
            //print_bodies_numbered(bi, len, DISTANCE_SCALE/40);
            
            totalmass = 0.0;
            totalenergy_k = 0.0;
            for (int i = 0; i < len; i++) {
                totalmass += (double)bi[i].mass;
                totalenergy_k += (double)get_body_energy(&bi[i]);
            }

            totalenergy_p = 0.0;
            for (int i = 0; i < (len - 1); i++) { 
                for (int j = (i+1); j < len; j++) {
                    totalenergy_p += (double)calculate_EG(&bi[i], &bi[j]);
                }
            }

            printf("Bodies:%d, System Mass:%e kg, Kinetic Energy:%e J, Potential Energy:%e J, Total Energy:%e J, Scale=%e meters, Tick=%lu\n", 
                    len, totalmass, totalenergy_k, totalenergy_p, (totalenergy_k - totalenergy_p), autoscale, tick);
            data[0] = tick; data[1] = len; data[2] = totalmass; data[3] = totalenergy_p; data[4] = (totalenergy_k - totalenergy_p); data[5] = autoscale;    
            writeCSV(6, data, fp);
        }

        tick++;
    }
    stopTime(&timer);
    
    printf("----=====+++ SIMULATION RESULTS +++=====----\n");
    printf("Elapsed Time: %f s | ",elapsedTime(timer));  printf("Seed: %d | ", seedNum); printf("Initial # Bodies: %d\n", numBodies); 
    printf("Max Ticks: %lu | ", max_ticks); printf("Seconds/Tick: %f | ", secs_per_tick); printf("Ticks/Display: %d\n", ticks_per_display);
    //double *data;
    
      
    //writeCSV(1, (int*) &numBodies, fp);
     
    printf("haha lmao\n");
}
