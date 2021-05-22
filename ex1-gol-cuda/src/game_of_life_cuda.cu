#include "common.h"

// CUDA error checking - derived from [https://stackoverflow.com/a/14038590]
#define cuda_error_check(x) {cuda_examine(x, __FILE__, __LINE__);}
inline void cuda_examine(cudaError_t code, const char * file, int line, \
                         bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "cuda_error_check: (%s:%d) %s\n", file, line, \
            cudaGetErrorString(code));

    if (abort)
    {
      exit(code);
    }
  }
}

// debug flags
// - `debug_verbose != 0` will annotate, to stderr, the program as it is
//   executed
// - `debug_timing != 0` will annotate, to stderr, the timing variables as
//   they are calculated
// - `debug_visual != 0` will annotate, to stderr, the ascii visualisation of
//   grid variables as they are intialised and updated
const int debug_verbose = 1;
const int debug_timing = 1;
const int debug_visual = 1;

// verbose macro
#define verbose(format, ...)                                \
  if (debug_verbose) {                                      \
    fprintf(stderr, "[verbose] "format"\n", ##__VA_ARGS__); \
  }

// timing macro
#define timing(format, ...)                                 \
  if (debug_timing) {                                       \
    fprintf(stderr, "[timing] "format"\n", ##__VA_ARGS__);  \
  }

// visual macro
#define visual(current_step, grid, n, m, format, ...)       \
  if (debug_visual) {                                       \
    fprintf(stderr, "[visual] "format"\n", ##__VA_ARGS__);  \
    visualise_ascii(current_step, grid, n, m);              \
  }

__global__ void gpu_game_of_life_step(int *current_grid, int *next_grid, \
                                      int n, int m)
{
  // indexing variables
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  int i = idx / m;
  int j = idx % m;

  // only perform kernel for valid cell indexes
  if ((i < n) && (j < m))
  {
    // neighbourhood variables
    int neighbours;
    int n_i[8], n_j[8];

    // count the number of neighbours, clockwise around the current cell.
    neighbours = 0;
    n_i[0] = i - 1; n_j[0] = j - 1;
    n_i[1] = i - 1; n_j[1] = j;
    n_i[2] = i - 1; n_j[2] = j + 1;
    n_i[3] = i;     n_j[3] = j + 1;
    n_i[4] = i + 1; n_j[4] = j + 1;
    n_i[5] = i + 1; n_j[5] = j;
    n_i[6] = i + 1; n_j[6] = j - 1;
    n_i[7] = i;     n_j[7] = j - 1;

    if (n_i[0] >= 0 && n_j[0] >= 0                                    \
        && current_grid[n_i[0] * m + n_j[0]] == ALIVE) neighbours++;
    if (n_i[1] >= 0                                                   \
        && current_grid[n_i[1] * m + n_j[1]] == ALIVE) neighbours++;
    if (n_i[2] >= 0 && n_j[2] < m                                     \
        && current_grid[n_i[2] * m + n_j[2]] == ALIVE) neighbours++;
    if (n_j[3] < m                                                    \
        && current_grid[n_i[3] * m + n_j[3]] == ALIVE) neighbours++;
    if (n_i[4] < n && n_j[4] < m                                      \
        && current_grid[n_i[4] * m + n_j[4]] == ALIVE) neighbours++;
    if (n_i[5] < n                                                    \
        && current_grid[n_i[5] * m + n_j[5]] == ALIVE) neighbours++;
    if (n_i[6] < n && n_j[6] >= 0                                     \
        && current_grid[n_i[6] * m + n_j[6]] == ALIVE) neighbours++;
    if (n_j[7] >= 0                                                   \
        && current_grid[n_i[7] * m + n_j[7]] == ALIVE) neighbours++;

    if (current_grid[i*m + j] == ALIVE && (neighbours == 2 || neighbours == 3))
    {
      next_grid[i*m + j] = ALIVE;
    }
    else if (current_grid[i*m + j] == DEAD && neighbours == 3)
    {
      next_grid[i*m + j] = ALIVE;
    }
    else
    {
      next_grid[i*m + j] = DEAD;
    }
  }
}

/*
  Implements the game of life on a grid of size `n` times `m`, starting from
  the `initial_state` configuration.

  If `nsteps` is positive, returns the last state reached.
*/
int* gpu_game_of_life(const int *initial_state, int n, int m, int nsteps, \
                      float *kernel_time)
{
  // cuda kernel parameters - uses least amount of blocks required
  const int n_threads = 1024;
  const int n_blocks = ((n * m - 1) / n_threads) + 1;

  verbose ("CUDA: <n_blocks> = %i, <n_threads> = %i", n_blocks, n_threads);

  // allocate gpu memory
  int *grid;
  int *updated_grid;

  cuda_error_check(cudaMalloc(&grid, sizeof(int) * n * m));
  cuda_error_check(cudaMalloc(&updated_grid, sizeof(int) * n * m));

  verbose ("CUDA: <grid>, <updated_grid> memory allocated (GPU)");

  // copy initial state to gpu memory
  cuda_error_check(cudaMemcpy(grid, initial_state, sizeof(int) * n * m, \
                              cudaMemcpyHostToDevice));

  verbose ("CUDA: copied <intial_state> (CPU) to <grid> (GPU)");

  // prepare kernel timing variables
  *kernel_time = 0.0;

  cudaEvent_t kernel_start, kernel_stop;
  cuda_error_check(cudaEventCreate(&kernel_start));
  cuda_error_check(cudaEventCreate(&kernel_stop));
  float kernel_time_step = 0.0;

  verbose ("CUDA: <kernel_start>, <kernel_stop> CUDA events defined");

  // initialise game_of_life loop
  int current_step = 0;

  while (current_step != nsteps)
  {
    current_step++;

    verbose("CUDA: <%i> GOL step started", current_step);

    // initialise timing of kernel execution
    cuda_error_check(cudaEventRecord(kernel_start));

    verbose("CUDA: <%i> timing intialised", current_step);

    // calculate next state of GOL using CUDA kernel across grid
    gpu_game_of_life_step<<<n_blocks, n_threads>>>(grid, updated_grid, n, m);

    verbose("CUDA: <%i> next GOL state calculated", current_step);

    // finalise timing of kernel execution
    cuda_error_check(cudaEventRecord(kernel_stop));
    cuda_error_check(cudaDeviceSynchronize(kernel_stop));

    // swap current and updated grid
    {
      int *tmp = grid;
      grid = updated_grid;
      updated_grid = tmp;
    }

    verbose("CUDA: <%i> grids swapped", current_step);

    // calculate timing of kernel execution
    cuda_error_check(cudaEventElapsedTime(&kernel_time_step, kernel_start, \
                                          kernel_stop));
    *kernel_time += kernel_time_step;

    timing("CUDA: <step_time, %i> = %f [ms]", current_step, kernel_time_step);

    // debug: visualise `grid` after current step
    if (debug_visual)
    {
      visual(current_step, grid, n, m, "<grid, %i> = ", current_step);
    }

    verbose("CUDA: <%i> GOL step finished", current_step);
  }

  verbose("CUDA: GOL loop finished");

  // copy final state to cpu memory
  int *final_state = (int *) malloc(sizeof(int) * n * m);

  if (final_state == NULL)
  {
    fprintf(stderr, "error while allocating memory for <final_state>\n");
    exit(1);
  }

  cuda_error_check(cudaMemcpy(final_state, grid, sizeof(int) * n * m, \
                              cudaMemcpyDeviceToHost));

  verbose ("CUDA: copied <grid> (GPU) to <final_state> (CPU)");

  // free gpu memory
  cudaFree(updated_grid);
  cudaFree(grid);

  verbose("CUDA: <grid>, <updated_grid> memory freed (GPU)");

  return final_state;
}

// write timing data for gpu CUDA code to file
int gpu_write_timing(struct Options const * opt, float const elapsed_time, \
                     float const kernel_time)
{
  FILE *file = NULL;
  char filename[200];
  int ierr = 0;

  // create filename for given options
  sprintf(filename, "output/timing-gpu-cuda.n-%i.m-%i.nsteps-%i.txt",  \
          opt->n, opt->m, opt->nsteps);

  printf("writing gpu timing data to filename: %s\n", filename);

  // open file
  file = fopen(filename, "w");

  if (file == NULL)
  {
    fprintf(stderr, "cannot open filename: %s\n", filename);
    ierr = 1;
  }
  else
  {
    // write timing data
    fprintf(file, "# gpu_elapsed_time, gpu_kernel_time\n");
    fprintf(file, "# [ms], [ms]\n");
    fprintf(file, "%f, %f\n", elapsed_time, kernel_time);

    // close file
    fclose(file);
  }

  return ierr;
}

// do not define the main function if this file is included somewhere else.
#ifndef INCLUDE_GPU_VERSION
int main(int argc, char **argv)
{
  // debug: verbose
  verbose("<debug_verbose> = on");
  if (debug_timing) verbose("<debug_timing> = on");
  if (debug_visual) verbose("<debug_visual> = on");

  // define timing variables
  struct timeval start;
  struct timeval gol_start;

  // initialise timing of entire program execution
  start = init_time();

  verbose("program timing initialised");

  // read input
  struct Options *opt = (struct Options *) malloc(sizeof(struct Options));
  getinput(argc, argv, opt);

  verbose("read input");

  // define parameter variables
  const int n = opt->n;
  const int m = opt->m;
  const int nsteps = opt->nsteps;

  verbose("parameters defined: <n> = %i, <m> = %i, <nsteps> = %i", \
          n, m, nsteps);

  // allocate memory for `initial_state` variable
  int *initial_state = (int *) malloc(sizeof(int) * n * m);

  if (initial_state == NULL)
  {
    fprintf(stderr, "error while allocating memory for <initial_state>\n");
    return -1;
  }

  verbose("<initial_state> memory allocated: sizeof(int) * %i", n * m);

  // generate initial conditions
  generate_IC(opt->iictype, initial_state, n, m);

  verbose("<initial_state> initial conditions generated");

  // debug: visualise `intial_state` after initial conditions
  visual(0, initial_state, n, m, "<initial_state> = ");

  // initialise timing of GOL simulation
  gol_start = init_time();

  verbose("GOL simulation timing initialised");

  // calculate `final_state` (and record kernel time)
  float kernel_time = 0.0;
  int *final_state = gpu_game_of_life(initial_state, n, m, nsteps,
                                      &kernel_time);

  // calculate time for GOL simulation
  float elapsed_time = get_elapsed_time(gol_start);
  timing("<elapsed_time> = %f [ms]", elapsed_time);

  // calculate kernel time
  timing("<kernel_time> = %f [ms] / (%f%%)", kernel_time,
         100.0*kernel_time/elapsed_time);

  verbose("GOL simulation timing finished");

  // write timing to file
  gpu_write_timing(opt, elapsed_time, kernel_time);

  verbose("<elapsed_time> written to file");

  // debug: visualise `final_state` after loop completion
  visual(nsteps, final_state, n, m, "<final_state> = ");

  // free cpu memory
  free(final_state);
  free(initial_state);
  free(opt);

  verbose("memory freed");

  // debug: calculate time for entire program execution
  float total_time = get_elapsed_time(start);
  timing("<total_time> = %f [ms]", total_time);

  verbose("program timing finished");

  return 0;
}
#endif
