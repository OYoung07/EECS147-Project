#include <stdio.h>
#include <stdlib.h>
#include "kernel.cu"
#include "support.h"
#include "body.h"

int main (int argc, char *argv[]) {
    Timer timer;
   
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
    b2.velocity.x = 0; //LEO
    b2.velocity.y = 7.8e3;
    b2.velocity.z = 0;
     
   
    printf("%.15f\n",calculate_FG(&b1,&b2));
   
    float3 vec = get_direction_vector(&b1,&b2);

    printf("(%.5f,%.5f,%.5f)\n",vec.x,vec.y,vec.z);

    float3 accel = get_accel_vector(&b1, &b2);

    print_float3(accel); printf("\n");

    struct body* bodies[2];
    const int len = 2;
    
    bodies[0] = &b1;
    bodies[1] = &b2;

    accel = CPU_reduce_accel_vectors(bodies[0], bodies, len);
    print_float3(accel); printf("\n");

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
