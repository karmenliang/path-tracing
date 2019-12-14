/*
 * Abstract material class with implementations for 
 * matte and metal surface materials
 */

#ifndef MATERIAL_H
#define MATERIAL_H

struct hit_record;

#include "ray.h"
#include "surface.h"

class material {
 public:

 /*
  * Produce a scattered ray
  */
  __device__ virtual bool scatter(const ray& r_in, const hit_record& rec,
				  vec3& attenuation, ray& scattered,
				  curandState *local_rand_state) const = 0;

};

#define RANDVEC3 vec3(curand_uniform(local_rand_state), curand_uniform(local_rand_state), curand_uniform(local_rand_state))

/* 
 * Randomly sample points in a unit radius centered at origin
 */
__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
    vec3 p;
    do {
        p = 2.0f*RANDVEC3 - vec3(1, 1, 1);
    } while (p.squared_length() >= 1.0f);
    return p;
}

////////////////////////////////////////////////////////////////////////////////////

/*
 * Diffuse material class:
 * Takes on the color of surroundings, modulated by its own color.
 * Reflected light has randomized direction.
 */
class matte : public material {
 public:

  // albedo: proportion of light reflected by surface
  __device__ matte(const vec3& a) : albedo(a) {}

  __device__ virtual bool scatter(const ray& r_in, const hit_record& rec,
				  vec3& attenuation, ray& scattered,
				  curandState *local_rand_state) const {

    vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
    scattered = ray(rec.p, target - rec.p);
    attenuation = albedo;

    return true;
  }

  vec3 albedo;

};


////////////////////////////////////////////////////////////////////////////////////

// Helper for metal class
__device__ vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2.0f * dot(v,n) * n;
}

/*
 * Metal material class:
 * Rays follow the formula in reflect helper.
 */
class metal : public material {
    public:
        __device__ metal(const vec3& a, float f) : albedo(a) {
            if (f < 1.0f) fuzz = f; else fuzz = 1.0f;
        }
  
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec,
					vec3& attenuation, ray& scattered,
					curandState *local_rand_state) const {

            vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere(local_rand_state));
            attenuation = albedo;

            return (dot(scattered.direction(), rec.normal) > 0.0f);
        }
	
        vec3 albedo;
  
	    // Fuzziness of reflections; 0 is perfect reflection
	    float fuzz;

};

#endif
