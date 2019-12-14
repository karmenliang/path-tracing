# Parallel Monte Carlo Path Tracing
CS338 Parallel Processing final project. A path tracer implemented sequentially and parallelized on the GPU with the CUDA computing platform.

Implementation based on Peter Shirley's [Ray Tracing in One Weekend](https://raytracing.github.io/books/RayTracingInOneWeekend.html).

### To do:

- [x] Construct large input scene to use for all testing
- [x] Collect benchmark performance data from sequential program
- [x] Collect performance data from 4 different block size configurations

### Extensions:

- [ ] Assign more computational work per thread
- [ ] Reduced branching
- [ ] Metropolis light transport
- [ ] Importance sampling
