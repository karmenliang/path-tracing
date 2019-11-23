#include <iostream>
#include <float.h>

#include "surface_list.h"
#include "sphere.h"
#include "camera.h"

#ifndef RANDOMH
#define RANDOMH

#include <cstdlib>

inline double random_double() {
    return rand() / (RAND_MAX + 1.0);
}

#endif

// Pick random pt in a unit radius centered at origin
vec3 random_in_unit_sphere() {
    vec3 p;
    do {
        p = 2.0*vec3(random_double(), random_double(), random_double()) - vec3(1,1,1);
    } while (p.squared_length() >= 1.0);
    return p;
}

// Simple gradient background scene
vec3 color(const ray& r, surface *world) {
  hit_record rec;
    if (world->hit(r, 0.001, FLT_MAX, rec)) {
      vec3 target = rec.p + rec.normal + random_in_unit_sphere();
      return 0.5 * color(ray(rec.p, target - rec.p), world);
    }
    
    else {
      vec3 unit_dir = unit_vector(r.direction());
      float t = 0.5*(unit_dir.y() + 1.0);
    
      return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
    }
}

// Slowly building up pieces of the path tracer
int main() {
    int nx = 200;
    int ny = 100;
    int ns = 100; // number of samples
    std::cout << "P3\n" << nx << " " << ny << "\n255\n";

    camera cam;

    surface *list[2];
    list[0] = new sphere(vec3(0,0,-1), 0.5);
    list[1] = new sphere(vec3(0,-100.5,-1), 100);
    surface *world = new surface_list(list,2);

    for (int j = ny-1; j >= 0; j--) {
        for (int i = 0; i < nx; i++) {
            vec3 col(0, 0, 0);

	    // Anti-aliasing
	    for (int s = 0; s < ns; s++) {
                float u = float(i + random_double()) / float(nx);
                float v = float(j + random_double()) / float(ny);
                ray r = cam.get_ray(u, v);
                col += color(r, world);
            }
	    
            col /= float(ns);
	    col = vec3( sqrt(col[0]), sqrt(col[1]), sqrt(col[2]) );

            int ir = int(255.99*col[0]);
            int ig = int(255.99*col[1]);
            int ib = int(255.99*col[2]);
            
            std::cout << ir << " " << ig << " " << ib << "\n";
        }
    }

}
