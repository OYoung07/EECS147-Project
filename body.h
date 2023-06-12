#ifndef __BODY_H__
#define __BODY_H__

#include <cuda_runtime.h>

struct body {
    int id;
    double mass;
    double radius;
    double3 position;
    double3 velocity;
};

double distance(struct body* b1, struct body* b2);
double calculate_FG(struct body* b1, struct body* b2);
double calculate_EG(struct body* b1, struct body* b2); 
double3 get_direction_vector(struct body* origin, struct body* actor);
double3 get_accel_vector(struct body* origin, struct body* actor);
double magnitude(double3 v);
double get_body_energy(struct body* b);
double3 CPU_reduce_accel_vectors(struct body b, struct body* bodies, const int &num_bodies);
__device__ __host__ struct body create_new_body(struct body* a, struct body* b); 
__device__ __host__ unsigned int delete_body_id(unsigned int id, struct body* bodies, const int &num_bodies);
struct body* get_body(unsigned int id, struct body* bodies, const int &num_bodies); 
 
void CPU_tick(struct body* bodies, const int &num_bodies, const double &t);

__device__ __host__ double3 get_barycenter(struct body* a, struct body* b); 
unsigned int CPU_collisions(struct body* bodies, int num_bodies); 

void print_bodies(struct body* bodies, const int &num_bodies, const double &tile_scale); 
void print_bodies_numbered(struct body* bodies, const int &num_bodies, const double &tile_scale); 
void print_body(struct body* b); 

__device__ __host__ double3 operator+(const double3 &a, const double3 &b); 
__device__ __host__ double3 operator-(const double3 &a, const double3 &b);
__device__ __host__ double3 operator*(const double3 &a, const double &b);
__device__ __host__ double3 operator/(const double3 &a, const double &b);

void print_double3(const double3 &f);

#endif
