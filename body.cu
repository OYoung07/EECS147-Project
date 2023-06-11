#include "body.h"
#include <math.h>
#include <stdio.h>

#define MAX_BODIES 256

/* operator overloading for float3 */
__device__ __host__  float3 operator+(const float3 &a, const float3 &b) {
    float3 c;

    c.x = a.x + b.x; 
    c.y = a.y + b.y; 
    c.z = a.z + b.z;

    return c;
}

__device__  __host__ float3 operator-(const float3 &a, const float3 &b) {
    float3 c;

    c.x = a.x - b.x;
    c.y = a.y - b.y;
    c.z = a.z - b.z;

    return c;
}

__device__  __host__ float3 operator*(const float3 &a, const float &b) {
    float3 c;

    c.x = a.x * b;
    c.y = a.y * b;
    c.z = a.z * b;

    return c;
}

__device__  __host__ float3 operator/(const float3 &a, const float &b) {
    float3 c;
    
    c.x = a.x / b;
    c.y = a.y / b;
    c.z = a.z / b;

    return c;
}

void print_float3(const float3 &f) {
    printf("(%e,%e,%e)", f.x, f.y, f.z);
}

void print_body(struct body* b) {
    printf("Body ID:%d\n",b->id);
    printf("Mass:%e\n",b->mass);
    printf("Radius:%e\n",b->radius);
    printf("Position:");
    print_float3(b->position);
    printf("\nVelocity:");
    print_float3(b->velocity);
    printf("\n");
}

//get distance between two bodies
float distance(struct body* b1, struct body* b2) {
    return sqrt(pow(b2->position.x - b1->position.x, 2) + 
                pow(b2->position.y - b1->position.y, 2) + 
                pow(b2->position.z - b1->position.z, 2));
}

//get gravity force magnitude between two bodies
float calculate_FG(struct body* b1, struct body* b2) {
    double G = 6.674e-11;
    double d = distance(b1, b2);
    double mag_F; 

    mag_F = (G * (double)b1->mass *(double)b2->mass)/pow(d, 2); //gravity formula

    return (float)mag_F;
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

//calculate mean acceleration vector from all other bodies
float3 CPU_reduce_accel_vectors(struct body b, struct body* bodies, const int &num_bodies) {
    float3 accel;
    accel.x = 0;
    accel.y = 0;
    accel.z = 0;    

    for (int i = 0; i < num_bodies; i++) {
        if (bodies[i].id != b.id) { //if not self
           accel = accel + get_accel_vector(&b, &bodies[i]); 
        }
    }

    return accel;
}

//collide two bodies inelasticly and return the resultant body
__device__ __host__ struct body create_new_body(struct body* a, struct body* b) {
    struct body c;

    c.mass = a->mass + b->mass;
    c.radius = cbrt((pow(a->radius,3) +  pow(b->radius,3))); //combine radii and conserve area

    if (a->mass >= b->mass) {
        c.position = a->position;
    } else {
        c.position = b->position;
    }

    c.velocity = ((a->velocity * a->mass) + (b->velocity * b->mass))/c.mass;

    return c;
}

__device__ __host__ unsigned int delete_body_id(unsigned int id, struct body* bodies, const int &num_bodies) {
    unsigned int delete_index = num_bodies;

    for (int i = 0; i < num_bodies; i++) {
        if (bodies[i].id = id) {
            delete_index = i;
            break;
        }
    }

    if (delete_index < (num_bodies - 1)) {
        for (int i = delete_index; i < (num_bodies - 1); i++) {
            bodies[i] = bodies[i+1];
        }
        return num_bodies - 1;
    } else if (delete_index == (num_bodies - 1)) {
        return num_bodies - 1;
    }

    return num_bodies;    
}

unsigned int CPU_collisions(struct body* bodies, int num_bodies) {
    unsigned int marked_for_death[MAX_BODIES];
    unsigned int death_index = 0;

    struct body new_bodies[10];
    unsigned int new_body_index = 0;

    unsigned int temp_id;    

    for (int i = (num_bodies - 1); i >= 0; i--) {
        for(int j = 0; j < num_bodies; j++) {
            for (int k = 0; k < death_index; k++) {
                if (marked_for_death[k] = bodies[i].id) {
                    num_bodies = delete_body_id(bodies[i].id, bodies, num_bodies);
                    break;
                }
            }

            if ((distance(&bodies[i], &bodies[j]) < (bodies[i].radius + bodies[j].radius)) && (i != j)) {
                marked_for_death[death_index] = bodies[j].id;
                death_index++;
                
                new_bodies[new_body_index] = create_new_body(&bodies[i], &bodies[j]);
                new_bodies[new_body_index].id = bodies[i].id; //we know it will be unused
                new_body_index++;
                
                num_bodies = delete_body_id(bodies[i].id, bodies, num_bodies);
                break;              
            }
        }
    }

    for (int i = 0; i < new_body_index; i++) {
        bodies[num_bodies] = new_bodies[i];
        num_bodies++;   
    }

    return num_bodies;
}

void CPU_tick(struct body* bodies, const int &num_bodies, const float &t) {
    float3 a;    

    /* allocate temp array for calculation */
    struct body* temp_bodies;
    temp_bodies = (struct body*) malloc(num_bodies * sizeof(struct body));
    memcpy(temp_bodies, bodies, (num_bodies * sizeof(struct body)));

    for (int i = 0; i < num_bodies; i++) {
        a = CPU_reduce_accel_vectors(bodies[i], temp_bodies, num_bodies);
        
        bodies[i].velocity = bodies[i].velocity + (a * (t/2.0)); //kick        
        bodies[i].position = bodies[i].position + (bodies[i].velocity * t); //drift
       
        a = CPU_reduce_accel_vectors(bodies[i], temp_bodies, num_bodies);

        bodies[i].velocity = bodies[i].velocity + (a * (t/2.0)); //kick 
    }

    free(temp_bodies); //memory leaks are bad
}

void print_bodies(struct body* bodies, const int &num_bodies, const float &tile_scale) {
    char map[40][40];
    float y_index;
    float x_index;    

    //draw true to size
    for (int y = 0; y < 40; y++) {
        for (int x = 0; x < 40; x++) {
            map[y][x] = ' ';
            for (int i = 0; i < num_bodies; i++) {
                if (sqrt(pow(bodies[i].position.x - ((x-20.0) * tile_scale), 2.0) + pow(bodies[i].position.y - ((y-20.0) * tile_scale), 2.0)) < bodies[i].radius) {
                    map[y][x] = '@';
                }  
            }
        }     
    }

    //draw as point mass if too small
    for (int i = 0; i < num_bodies; i++) {
        y_index = (bodies[i].position.y / tile_scale) + 20;
        x_index = (bodies[i].position.x / tile_scale) + 20;
        
        if (y_index < 40 && y_index >= 0 && x_index < 40 && x_index >= 0) {
            if (map[(int)y_index][(int)x_index] != '@') {
                map[(int)y_index][(int)x_index] = '.';
            }
        }
    }

    //print
    printf("\e[1;1H\e[2J"); //clear screen
    for (int y = 0; y < 40; y++) {
        for (int x = 0; x < 40; x++) {
            printf(" %c",map[y][x]);
        }
        printf("\n");
    }
}

void print_bodies_numbered(struct body* bodies, const int &num_bodies, const float &tile_scale) {
    char map[40][40];
    int y_index;
    int x_index;    

    for (int y = 0; y < 40; y++) {
        for (int x = 0; x < 40; x++) {
            map[y][x] = ' ';
        }
    }

    for (int i = 0; i < num_bodies; i++) {
        y_index = (int)(bodies[i].position.y / tile_scale) + 20;
        x_index = (int)(bodies[i].position.x / tile_scale) + 20;
        
        if (y_index < 40 && y_index >= 0 && x_index < 40 && x_index >= 0) {
            map[y_index][x_index] = ((char)bodies[i].id % 90) + 33; //give unique character
        }
    }

    //print
    printf("\e[1;1H\e[2J"); //clear screen
    for (int y = 0; y < 40; y++) {
        for (int x = 0; x < 40; x++) {
            printf(" %c",map[y][x]);
        }
        printf("\n");
    }
}
