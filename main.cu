#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.h"
#include "body.h"

int main (int argc, char *argv[]) {
    Timer timer;
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
