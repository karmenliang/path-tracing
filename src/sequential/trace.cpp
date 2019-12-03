/*
 * Sequential path tracer using the Monte Carlo method.
 */

#include <iostream>
#include <float.h>

#include "geometry/surface_list.h"
#include "geometry/sphere.h"
#include "material.h"
#include "camera.h"
#include "jpeg.h"

/*
 * Determine the color of a point by backward tracing
 */
vec3 color(const ray& r, surface *world, int depth) {
  hit_record rec;
  
  if (world->hit(r, 0.001, FLT_MAX, rec)) {

    ray scattered;
    vec3 attenuation;

    if (depth < 50 && rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
      return attenuation*color(scattered, world, depth+1);
    }else {
      return vec3(0,0,0);
    }
    
  }else {

    vec3 unit_dir = unit_vector(r.direction());
    float t = 0.5*(unit_dir.y() + 1.0);
    
    return (1.0 - t) * vec3(1.0, 1.0, 1.0) + t * vec3(0.5, 0.7, 1.0);
  }
}

/*
 * Generate a random scene with n spheres
 */
surface *random_scene(int n) {

  surface **list = new surface*[n+1];
  list[0] =  new sphere(vec3(0,-1000,0), 1000, new matte(vec3(0.5, 0.5, 0.5)));
  int i = 1;

  for (int a = -3; a < 4; a++) {
    for (int b = -3; b < 4; b++) {

      float choose_mat = random_double();
      vec3 center(a+0.9*random_double(),0.2,b+0.9*random_double());

      if ((center-vec3(4,0.2,0)).length() > 0.9) {

	if (choose_mat < 0.5) {  // diffuse
	  list[i++] = new sphere(center, 0.2,
				 new matte(vec3(random_double()*random_double(),
						random_double()*random_double(),
						random_double()*random_double())
					   )
				 );
	}

	else { // metal
	  list[i++] = new sphere(center, 0.2,
				 new metal(vec3(0.5*(1 + random_double()),
						0.5*(1 + random_double()),
						0.5*(1 + random_double())),
					   0.5*random_double()));
	}
      }
    }
  }

  list[i++] = new sphere(vec3(-2, 1, 0), 1.0, new matte(vec3(0.4, 0.2, 0.1)));
  list[i++] = new sphere(vec3(2, 1, 0), 1.0, new metal(vec3(0.7, 0.6, 0.5), 0.0));

  return new surface_list(list,i);
}

int main() {
  int nx = 600; // image width
  int ny = 300; // image height
  int ns = 150; // number of samples
  std::cout << "P3\n" << nx << " " << ny << "\n255\n";

  camera cam(vec3(6,1,3), vec3(0,0,-1), vec3(0,1,0), 60, float(nx)/float(ny));
  surface *world = random_scene(50);

  for (int j = ny-1; j >= 0; j--) {

    for (int i = 0; i < nx; i++) {

      vec3 col(0, 0, 0);

      // Anti-aliasing
      for (int s = 0; s < ns; s++) {
	float u = float(i + random_double()) / float(nx);
	float v = float(j + random_double()) / float(ny);
	ray r = cam.get_ray(u, v);
	col += color(r, world, 0);
      }
	    
      col /= float(ns);
      col = vec3( sqrt(col[0]), sqrt(col[1]), sqrt(col[2]) );

      int ir = int(255.99*col[0]);
      int ig = int(255.99*col[1]);
      int ib = int(255.99*col[2]);

      // Writing out RGB values to ppm file
      std::cout << ir << " " << ig << " " << ib << "\n";
    }
  }

}
