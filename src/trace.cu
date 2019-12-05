/*
 * Parallel path tracer with CUDA.
 */

#include <iostream>
#include <float.h>

// CUDA libraries
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include <curand_kernel.h>

#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "surface_list.h"
#include "camera.h"

///////////////////////////////////////////////////////////////////////////////////

__device__ vec3 color(const ray& r, hittable **world) {
  hit_record rec;

  if ((*world)->hit(r, 0.0, FLT_MAX, rec)) {
    return 0.5f*vec3(rec.normal.x()+1.0f, rec.normal.y()+1.0f, rec.normal.z()+1.0f);
    
  }else {
    vec3 unit_direction = unit_vector(r.direction());
    float t = 0.5f*(unit_direction.y() + 1.0f);

    return (1.0f-t)*vec3(1.0, 1.0, 1.0) + t*vec3(0.5, 0.7, 1.0);
  }
}

/*
 * CUDA kernel: initialize rand_state, separated from 
 * actual rendering for performance measurement
 */
__global__ void render_init(int max_x, int max_y, curandState *rand_state) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if((i >= max_x) || (j >= max_y)) return;
  int pixel_index = j*max_x + i;

  // Each thread gets same seed, a different sequence number, no offset
  curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}


/*
 * CUDA kernel function
 */
__global__ void render(vec3 *fb, int max_x, int max_y, int ns, camera **cam,
		       hittable **world, curandState *rand_state) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  // Only run on pixels within the image
  if((i >= max_x) || (j >= max_y)) return;

  int pixel_index = j*max_x + i;
  vec3 col(0, 0, 0);
  
  // Local copy of random state
  curandState local_rand_state = rand_state[pixel_index];
  
  for (int s = 0; s < ns; s++){
    float u = float(i + curand_uniform(&local_rand_state)) / float(max_x);
    float v = float(j + curand_uniform(&local_rand_state)) / float(max_y);

    ray r = (*cam)->get_ray(u,v);
    col += color(r, world);
  }
  
  fb[pixel_index] = col/float(ns);
}

/*
 * CUDA kernel: construct scene's objects
 */
__global__ void create_world(hittable **d_list, hittable **d_world, camera **d_camera) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *(d_list)   = new sphere(vec3(0,0,-1), 0.5);
    *(d_list+1) = new sphere(vec3(0,-100.5,-1), 100);
    *d_world    = new surface_list(d_list,2);
    *d_camera   = new camera();
  }
}

/*
 * CUDA kernel: deallocate scene's objects
 */
__global__ void free_world(hittable **d_list, hittable **d_world, camera **d_camera) {
  delete *(d_list);
  delete *(d_list+1);
  delete *d_world;
  delete *d_camera;
}

/********************* MAIN ******************************************************/

int main() {

  int nx = 1200; // image width
  int ny = 600;  // image height
  int tx = 8;    // block width
  int ty = 8;    // block height
  int ns = 100;  // number of samples
  
  int num_pixels = nx*ny;
  size_t fb_size = num_pixels*sizeof(vec3);
  
  std::cerr << "--------------------------------------------------------------\n\n";
  std::cerr << "Rendering a " << nx << "x" << ny << " image with "
	    << ns << "samples per pixel. \n";
  std::cerr << tx << "x" << ty << " blocks.\n";

  // Allocate unified memory for frame buffer
  vec3 *fb;
  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

  // Allocate random state
  curandState *d_rand_state;
  checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));
  
  // Allocate world of objects and the camera
  hittable **d_list;
  checkCudaErrors(cudaMalloc((void **)&d_list, 2*sizeof(hittable *)));
  hittable **d_world;
  checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(hittable *)));
  camera **d_camera;
  checkCudaErrors(cudaMalloc((void **)&d_camera, sizeof(camera *)));
  create_world<<<1,1>>>(d_list, d_world, d_camera);

  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  
  // For timing execution of kernel code
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  // Dimensions of grids and blocks
  dim3 dimGrid(nx/tx+1,ny/ty+1);
  dim3 dimBlock(tx,ty);

  render_init<<<dimGrid, dimBlock>>>(nx, ny, d_rand_state);
  
  checkCudaErrors(cudaEventRecord(start));
  
  // Kernel invocation
  render<<<dimGrid, dimBlock>>>(fb, nx, ny, ns, d_camera, d_world, d_rand_state);

  checkCudaErrors(cudaGetLastError());

  // Wait for GPU to finish
  checkCudaErrors(cudaDeviceSynchronize());
  
  // Print out performance metrics
  checkCudaErrors(cudaEventRecord(stop));
  checkCudaErrors(cudaEventSynchronize(stop));
  float ms = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&ms, start, stop));
  std::cerr << "PERFORMANCE: " << ms << " ms\n\n";
  std::cerr << "--------------------------------------------------------------\n";

  // Output frame buffer as image
  std::cout << "P3\n" << nx << " " << ny << "\n255\n";
  for (int j = ny-1; j >= 0; j--) {
    for (int i = 0; i < nx; i++) {

      size_t pixel_index = j*nx + i;

      int ir = int(255.99*fb[pixel_index].r());
      int ig = int(255.99*fb[pixel_index].g());
      int ib = int(255.99*fb[pixel_index].b());

      std::cout << ir << " " << ig << " " << ib << "\n";

    }
  }

  // CUDA clean-up
  checkCudaErrors(cudaDeviceSynchronize());
  free_world<<<1, 1>>>(d_list,d_world, d_camera);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(d_camera));
  checkCudaErrors(cudaFree(d_rand_state));
  checkCudaErrors(cudaFree(fb));
  
}
