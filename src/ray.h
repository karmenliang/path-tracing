#ifndef RAY_H
#define RAY_H

#include "vec3.h"

class ray {
  
 public:

  __device__ ray() {}

  __device__ ray(const vec3& a, const vec3& b) {
    A = a; // ray origin
    B = b; // ray direction
  }

  __device__ vec3 origin() const       { return A; }
  __device__ vec3 direction() const    { return B; }

  // t: value that changes 'destination' along ray
  __device__ vec3 point_at_parameter(float t) const { return A + t*B; }

  vec3 A;
  vec3 B;

};

#endif
