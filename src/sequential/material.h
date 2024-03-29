/*
 * Abstract material class with implementations for 
 * matte and metal surface materials
 */

#ifndef MATERIAL_H
#define MATERIAL_H

#include <cstdlib>

class material  {
 public:

 /*
  * Produce a scattered ray
  */
  virtual bool scatter(const ray& r_in, const hit_record& rec,
		       vec3& attenuation, ray& scattered) const = 0;

};

// Random util
inline double random_double() {
  return rand() / (RAND_MAX + 1.0);
}

/* 
 * Randomly sample points in a unit radius centered at origin
 */
vec3 random_in_unit_sphere() {
  vec3 p;
    
  do {
      p = 2.0*vec3(random_double(), random_double(), random_double()) - vec3(1,1,1);
  } while (p.squared_length() >= 1.0);
  
  return p;
}

/*
 * Diffuse material class:
 * Takes on the color of surroundings, modulated by its own color.
 * Reflected light has randomized direction.
 */
class matte : public material {
 public:

 // albedo: proportion of light reflected by surface
 matte(const vec3& a) : albedo(a) {}

  virtual bool scatter(const ray& r_in, const hit_record& rec,
		       vec3& attenuation, ray& scattered) const {

    vec3 target = rec.p + rec.normal + random_in_unit_sphere();
    scattered = ray(rec.p, target-rec.p);
    attenuation = albedo;

    return true;
  }

  vec3 albedo;
};

// Helper for metal class
vec3 reflect(const vec3& v, const vec3& n) {
    return v - 2*dot(v,n)*n;
}

/*
 * Metal material class:
 * Rays follow the formula in reflect helper.
 */
class metal : public material {
    public:
        metal(const vec3& a, float f) : albedo(a) {
            if (f < 1) fuzz = f; else fuzz = 1;
        }
  
        virtual bool scatter(const ray& r_in, const hit_record& rec,
                             vec3& attenuation, ray& scattered) const {

            vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere());
            attenuation = albedo;

            return (dot(scattered.direction(), rec.normal) > 0);
        }
	
        vec3 albedo;
  
  // Fuzziness of reflections; 0 is perfect reflection
	float fuzz;
};

#endif