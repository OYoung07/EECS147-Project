#ifndef __BODY_H__
#define __BODY_H__

#include <cuda_runtime.h>

struct body {
    float mass;
    float radius;
    float3 position;
    float3 velocity;
};

float distance(struct body* b1, struct body* b2);
float calculate_FG(struct body* b1, struct body* b2);
float3 get_direction_vector(struct body* origin, struct body* actor);


float3 operator+(const float3 &a, const float3 &b); 
float3 operator-(const float3 &a, const float3 &b);
float3 operator*(const float3 &a, const float &b);
float3 operator/(const float3 &a, const float &b);
#endif
