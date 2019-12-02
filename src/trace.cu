/*
 * Parallel path tracer with CUDA.
 */

#include <iostream>

// CUDA libraries
#include <cuda_runtime.h>
#include <helper_cuda.h>

#include "vec3.h"

/*
 * CUDA kernel function
 */
__global__ void render(vec3 *fb, int max_x, int max_y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;

  // Only run on pixels within the image
  if((i >= max_x) || (j >= max_y)) return;

  int pixel_index = j*max_x + i;
  fb[pixel_index] = vec3(float(i) / max_x,
			 float(j) / max_y,
			 0.2);
}

/********************* MAIN ******************************************************/

int main() {

  int nx = 1200; // image width
  int ny = 600;  // image height
  int tx = 8;    // block width
  int ty = 8;    // block height

  std::cerr << "--------------------------------------------------------------\n\n";
  std::cerr << "Rendering a " << nx << "x" << ny << " image ";
  std::cerr << "in " << tx << "x" << ty << " blocks.\n";

  int num_pixels = nx*ny;
  size_t fb_size = num_pixels*sizeof(vec3);

  // Allocate unified memory for frame buffer
  vec3 *fb;
  checkCudaErrors(cudaMallocManaged((void **)&fb, fb_size));

  // For timing execution of kernel code
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));

  // Dimensions of grids and blocks
  dim3 dimGrid(nx/tx+1,ny/ty+1);
  dim3 dimBlock(tx,ty);

  checkCudaErrors(cudaEventRecord(start));
  
  // Kernel invocation
  render<<<dimGrid, dimBlock>>>(fb, nx, ny);
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

  checkCudaErrors(cudaFree(fb));
}
