#include <stdlib.h>
#include <stdio.h>
#include "support.h"

void initVector(float **vec_h, unsigned size)
{
    *vec_h = (float*)malloc(size*sizeof(float));

    if(*vec_h == NULL) {
        FATAL("Unable to allocate host");
    }
    srand(217);
    for (unsigned int i=0; i < size; i++) {
        (*vec_h)[i] = (rand()%100)/100.00;
    }

}


void verify(float* input, unsigned num_elements, float result) {

  const float relativeTolerance = 2e-5;

  float sum = 0.0f;
  for(int i = 0; i < num_elements; ++i) {
    sum += input[i];
  }
  printf("\n Sum: %f \n", sum);
  float relativeError = (sum - result)/sum;
  if (relativeError > relativeTolerance
    || relativeError < -relativeTolerance) {
    printf("TEST FAILED, cpu = %0.3f, gpu = %0.3f\n\n", sum, result);
    exit(0);
  }
  printf("TEST PASSED\n\n");

}

void startTime(Timer* timer) {
    gettimeofday(&(timer->startTime), NULL);
}

void stopTime(Timer* timer) {
    gettimeofday(&(timer->endTime), NULL);
}

float elapsedTime(Timer timer) {
    return ((float) ((timer.endTime.tv_sec - timer.startTime.tv_sec) \
                + (timer.endTime.tv_usec - timer.startTime.tv_usec)/1.0e6));
}

unsigned long timerPrompt() {
    int userInput;
    unsigned long long timeLimit;

    printf("Enter Max Ticks for Runtime:\nOR Enter 0 for Unlimited Ticks.\n"); //Prompts user to enter maximum ticks
    scanf("%d", &userInput);

    if ("%d", userInput == 0) {
        timeLimit = 2000000000; //basically infinite integer, change type if needed
    }
    else {
        timeLimit = userInput; //user chosen value
    }

    return timeLimit; 
}
float tickTime() {
    unsigned long userInput;

    printf("Enter amount of ticks/second:\n");
    scanf("%d", &userInput);

    return userInput;
}
