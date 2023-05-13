#include "body.h"
#include <math.h>
#include <stdio.h>

/* operator overloading for float3 */
float3 operator+(const float3 &a, const float3 &b) {
    float3 c;

    c.x = a.x + b.x; 
    c.y = a.y + b.y; 
    c.z = a.z + b.z;

    return c;
}

float3 operator-(const float3 &a, const float3 &b) {
    float3 c;

    c.x = a.x - b.x;
    c.y = a.y - b.y;
    c.z = a.z - b.z;

    return c;
}

float3 operator*(const float3 &a, const float &b) {
    float3 c;

    c.x = a.x * b;
    c.y = a.y * b;
    c.z = a.z * b;

    return c;
}

float3 operator/(const float3 &a, const float &b) {
    float3 c;
    
    c.x = a.x / b;
    c.y = a.y / b;
    c.z = a.z / b;

    return c;
}

void print_float3(const float3 &f) {
    printf("(%e,%e,%e)", f.x, f.y, f.z);
}


//get distance between two bodies
float distance(struct body* b1, struct body* b2) {
    return sqrt(pow(b2->position.x - b1->position.x, 2) + 
                pow(b2->position.y - b1->position.y, 2) + 
                pow(b2->position.z - b1->position.z, 2));
}

//get gravity force magnitude between two bodies
float calculate_FG(struct body* b1, struct body* b2) {
    float G = 6.674e-11;
    float d = distance(b1, b2);
    float mag_F; 

    mag_F = (G * b1->mass * b2->mass)/pow(d, 2); //gravity formula

    return mag_F;
}

//get direction vector between two bodies
float3 get_direction_vector(struct body* origin, struct body* actor) {
    float3 direction;
    float norm = distance(origin, actor);

    direction = actor->position - origin->position;
    direction = direction / norm;

    return direction;
}

/* calculate acceleration of origin as exerted by actor */
float3 get_accel_vector(struct body* origin, struct body* actor) {
    float F = calculate_FG(origin, actor);
    float3 dir = get_direction_vector(origin, actor);

    float3 F_vec = dir * F; //get force vector
    float3 A_vec = F_vec / origin->mass; //F = MA -> A = F/M

    return A_vec;
}

void tick(struct body* b, const float3 &a, const float &t) {
          
}

