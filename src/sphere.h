#ifndef SPHERE_H
#define SPHERE_H

#include "surface.h"

class sphere: public hittable  {
 public:

  __device__ sphere() {}

  __device__ sphere(vec3 cen, float r, material *m) : center(cen), radius(r), mat_ptr(m) {};

  __device__ virtual bool hit(const ray& r, float tmin, float tmax,
			      hit_record& rec) const;

  vec3 center;

  float radius;

  material *mat_ptr;

};

__device__ bool sphere::hit(const ray& r, float t_min, float t_max,
			    hit_record& rec) const {
  
  vec3 oc = r.origin() - center;
  float a = dot(r.direction(), r.direction());
  float b = dot(oc, r.direction());
  float c = dot(oc, oc) - radius*radius;
  float discriminant = b*b - a*c;

  // If discriminant > 0, there is a hit
  if (discriminant > 0) {

    // First root
    float temp = (-b - sqrt(discriminant))/a;

    // Check if first root is within range
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = r.point_at_parameter(rec.t);
      rec.normal = (rec.p - center) / radius;
      rec.mat_ptr = mat_ptr;
      
      return true;
    }

    // Second root
    temp = (-b + sqrt(discriminant)) / a;

    // Check if second root is within range
    if (temp < t_max && temp > t_min) {
      rec.t = temp;
      rec.p = r.point_at_parameter(rec.t);
      rec.normal = (rec.p - center) / radius;
      rec.mat_ptr = mat_ptr;
      
      return true;
    }
  }

  // Otherwise, no hit
  return false;
}


#endif
