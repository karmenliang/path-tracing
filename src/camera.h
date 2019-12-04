/*
 * A positionable camera with:
 * vfov:     vertical field of view, in degrees
 * lookfrom: point camera is placed
 * lookat:   point camera is directed at
 * vup:      view up, (0,1,0) keeps camera level
 * aspect:   aspect ratio, x/y
 * TODO: make positionable
 */


#ifndef CAMERA_H
#define CAMERA_H

#include "ray.h"

class camera {
 public:
  
  __device__ camera() {
    lower_left_corner = vec3(-2.0, -1.0, -1.0);
    horizontal = vec3(4.0, 0.0, 0.0);
    vertical = vec3(0.0, 2.0, 0.0);
    origin = vec3(0.0, 0.0, 0.0);
  }

  __device__ ray get_ray(float u, float v) { return ray(origin, lower_left_corner + u*horizontal + v*vertical - origin); }

  vec3 origin;
  vec3 lower_left_corner;
  vec3 horizontal;
  vec3 vertical;
};

#endif
