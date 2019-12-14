#ifndef SURFACE_H
#define SURFACE_H

#include "ray.h"

class material;

// Args for material to determine how rays interact with surfaces
struct hit_record
{
    float t;
    vec3 p;
    vec3 normal;
    material *mat_ptr;
};

class hittable  {
    public:
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif
