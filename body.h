#ifndef __BODY_H__
#define __BODY_H__

#include <cuda_runtime.h>

struct body {
    int id;
    float mass;
    float radius;
    float3 position;
    float3 velocity;
};

float distance(struct body* b1, struct body* b2);
float calculate_FG(struct body* b1, struct body* b2);
float3 get_direction_vector(struct body* origin, struct body* actor);
float3 get_accel_vector(struct body* origin, struct body* actor);
float3 CPU_reduce_accel_vectors(struct body b, struct body* bodies, const int &num_bodies);
__device__ __host__ struct body create_new_body(struct body* a, struct body* b); 
__device__ __host__ unsigned int delete_body_id(unsigned int id, struct body* bodies, const int &num_bodies);
 
void CPU_tick(struct body* bodies, const int &num_bodies, const float &t);
unsigned int CPU_collisions(struct body* bodies, int num_bodies); 

void print_bodies(struct body* bodies, const int &num_bodies, const float &tile_scale); 
void print_bodies_numbered(struct body* bodies, const int &num_bodies, const float &tile_scale); 
void print_body(struct body* b); 

__device__ __host__ float3 operator+(const float3 &a, const float3 &b); 
__device__ __host__ float3 operator-(const float3 &a, const float3 &b);
__device__ __host__ float3 operator*(const float3 &a, const float &b);
__device__ __host__ float3 operator/(const float3 &a, const float &b);

void print_float3(const float3 &f);

#endif
