#include <body.h>
#include <math.h>

//get distance between two bodies
float dist(struct body* b1, struct body* b2) {
    return sqrt(pow(b2.position.x - b1.position.x, 2) + pow(b2.position.y - b1.position.y, 2) + pow(b2.position.z - b1.position.z, 2))
}

//get gravity force magnitude between two bodies
float calculate_FG(struct body* b1, struct body* b2) {
    float G = 6.674e-11;
    float d = dist(b1, b2);
    float mag_F   

    mag_F = (G * b1->mass * b2->mass)/pow(d, 2); //gravity formula

    return mag_F;
}
