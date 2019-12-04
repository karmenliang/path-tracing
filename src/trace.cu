/*
 * Parallel path tracer with CUDA.
 */

#include <iostream>
#include <float.h>

// CUDA libraries
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "vec3.h"
#include "ray.h"
#include "sphere.h"
#include "surface_list.h"

///////////////////////////////////////////////////////////////////////////////////

__device__ vec3 color(const ray& r, surface **world) {
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
 * CUDA kernel function
 */
__global__ void render(vec3 *fb, int max_x, int max_y, vec3 lower_left,
		       vec3 horizontal, vec3 vertical, vec3 origin, surface **world) {

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  // Only run on pixels within the image
  if((i >= max_x) || (j >= max_y)) return;

  int pixel_index = j*max_x + i;
  float u = float(i) / float(max_x);
  float v = float(j) / float(max_y);

  ray r(origin, lower_left + u*horizontal + v*vertical);
  fb[pixel_index] = color(r, world);
}

/*
 * CUDA kernel: construct scene's objects
 */
__global__ void create_world(surface **d_list, surface **d_world) {
  if (threadIdx.x == 0 && blockIdx.x == 0) {
    *(d_list)   = new sphere(vec3(0,0,-1), 0.5);
    *(d_list+1) = new sphere(vec3(0,-100.5,-1), 100);
    *d_world    = new hitable_list(d_list,2);
  }
}

/*
 * CUDA kernel: deallocate scene's objects
 */
__global__ void free_world(surface **d_list, surface **d_world) {
  delete *(d_list);
  delete *(d_list+1);
  delete *d_world;
}

/********************* MAIN ******************************************************/

int main() {

  int nx = 1200; // image width
  int ny = 600;  // image height
  int tx = 8;    // block width
  int ty = 8;    // block height

  int num_pixels = nx*ny;
  size_t fb_size = num_pixels*sizeof(vec3);
  
  std::cerr << "--------------------------------------------------------------\n\n";
  std::cerr << "Rendering a " << nx << "x" << ny << " image ";
  std::cerr << "in " << tx << "x" << ty << " blocks.\n";

  // Allocate unified memory for frame buffer
  vec3 *fb;
  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

  // Construct world of objects
  surface **d_list;
  checkCudaErrors(cudaMalloc((void **)&d_list, 2*sizeof(surface *)));
  surface **d_world;
  checkCudaErrors(cudaMalloc((void **)&d_world, sizeof(surface *)));
  create_world<<<1,1>>>(d_list,d_world);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaDeviceSynchronize());
  
  // For timing execution of kernel code
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  // Dimensions of grids and blocks
  dim3 dimGrid(nx/tx+1,ny/ty+1);
  dim3 dimBlock(tx,ty);

  checkCudaErrors(cudaEventRecord(start));
  
  // Kernel invocation
  render<<<dimGrid, dimBlock>>>(fb, nx, ny,
				vec3(-2.0, -1.0, -1.0), // lower left corner
				vec3(4.0, 0.0, 0.0),    // horizontal
				vec3(0.0, 2.0, 0.0),    // vertical
				vec3(0.0, 0.0, 0.0),    // origin
				d_world);

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
  free_world<<<1, 1>>>(d_list,d_world);
  checkCudaErrors(cudaGetLastError());
  checkCudaErrors(cudaFree(d_list));
  checkCudaErrors(cudaFree(d_world));
  checkCudaErrors(cudaFree(fb));
}
