#ifndef __BODY_H__
#define __BODY_H__

struct body {
    float mass;
    float radius;
    float3 position;
    float3 velocity;
};

float dist(struct body* b1, struct body* b2);
float calculate_FG(struct body* b1, struct body* b2);

#endif
