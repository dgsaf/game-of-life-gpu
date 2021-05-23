#include "common.h"

// count number of living neighbours around cell (i, j)
int gol_neighbours(int const * const restrict current_grid, int n, int m,
                   int i, int j)
{
  int neighbours;
  int n_i[8], n_j[8];

  // index neighbours of current cell, with clockwise orientation
  neighbours = 0;
  n_i[0] = i - 1; n_j[0] = j - 1;
  n_i[1] = i - 1; n_j[1] = j;
  n_i[2] = i - 1; n_j[2] = j + 1;
  n_i[3] = i;     n_j[3] = j + 1;
  n_i[4] = i + 1; n_j[4] = j + 1;
  n_i[5] = i + 1; n_j[5] = j;
  n_i[6] = i + 1; n_j[6] = j - 1;
  n_i[7] = i;     n_j[7] = j - 1;

  // count the number of living neighbours
  if (n_i[0] >= 0 && n_j[0] >= 0
      && current_grid[n_i[0] * m + n_j[0]] == ALIVE) neighbours++;
  if (n_i[1] >= 0
      && current_grid[n_i[1] * m + n_j[1]] == ALIVE) neighbours++;
  if (n_i[2] >= 0 && n_j[2] < m
      && current_grid[n_i[2] * m + n_j[2]] == ALIVE) neighbours++;
  if (n_j[3] < m
      && current_grid[n_i[3] * m + n_j[3]] == ALIVE) neighbours++;
  if (n_i[4] < n && n_j[4] < m
      && current_grid[n_i[4] * m + n_j[4]] == ALIVE) neighbours++;
  if (n_i[5] < n
      && current_grid[n_i[5] * m + n_j[5]] == ALIVE) neighbours++;
  if (n_i[6] < n && n_j[6] >= 0
      && current_grid[n_i[6] * m + n_j[6]] == ALIVE) neighbours++;
  if (n_j[7] >= 0
      && current_grid[n_i[7] * m + n_j[7]] == ALIVE) neighbours++;

  return neighbours;
}

// determine the next state, for a cell with a given state and number of living
// neighbours
int gol_update(int state, int neighbours)
{
  int next_state = DEAD;

  if (state == ALIVE && (neighbours == 2 || neighbours == 3))
  {
    next_state = ALIVE;
  }
  else if (state == DEAD && neighbours == 3)
  {
    next_state = ALIVE;
  }
  else
  {
    next_state = DEAD;
  }

  return next_state;
}

void game_of_life(struct Options *opt, int *current_grid, int *next_grid,
                  int n, int m)
{
  int neighbours;
  int n_i[8], n_j[8];
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < m; j++)
    {
      // index neighbours of current cell, with clockwise orientation
      neighbours = 0;
      n_i[0] = i - 1; n_j[0] = j - 1;
      n_i[1] = i - 1; n_j[1] = j;
      n_i[2] = i - 1; n_j[2] = j + 1;
      n_i[3] = i;     n_j[3] = j + 1;
      n_i[4] = i + 1; n_j[4] = j + 1;
      n_i[5] = i + 1; n_j[5] = j;
      n_i[6] = i + 1; n_j[6] = j - 1;
      n_i[7] = i;     n_j[7] = j - 1;

      // count the number of living neighbours
      if (n_i[0] >= 0 && n_j[0] >= 0
          && current_grid[n_i[0] * m + n_j[0]] == ALIVE) neighbours++;
      if (n_i[1] >= 0
          && current_grid[n_i[1] * m + n_j[1]] == ALIVE) neighbours++;
      if (n_i[2] >= 0 && n_j[2] < m
          && current_grid[n_i[2] * m + n_j[2]] == ALIVE) neighbours++;
      if (n_j[3] < m
          && current_grid[n_i[3] * m + n_j[3]] == ALIVE) neighbours++;
      if (n_i[4] < n && n_j[4] < m
          && current_grid[n_i[4] * m + n_j[4]] == ALIVE) neighbours++;
      if (n_i[5] < n
          && current_grid[n_i[5] * m + n_j[5]] == ALIVE) neighbours++;
      if (n_i[6] < n && n_j[6] >= 0
          && current_grid[n_i[6] * m + n_j[6]] == ALIVE) neighbours++;
      if (n_j[7] >= 0
          && current_grid[n_i[7] * m + n_j[7]] == ALIVE) neighbours++;

      if (current_grid[i*m + j] == ALIVE
          && (neighbours == 2 || neighbours == 3))
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
}

void game_of_life_stats(struct Options *opt, int step, int *current_grid)
{
  unsigned long long num_in_state[NUMSTATES];
  int m = opt->m;
  int n = opt->n;

  for (int i = 0; i < NUMSTATES; i++) num_in_state[i] = 0;

  for (int j = 0; j < m; j++)
  {
    for (int i = 0; i < n; i++)
    {
      num_in_state[current_grid[i*m + j]]++;
    }
  }

  double frac;
  double ntot = opt->m*opt->n;

  FILE *fptr;

  if (step == 0)
  {
    fptr = fopen(opt->statsfile, "w");
  }
  else
  {
    fptr = fopen(opt->statsfile, "a");
  }

  fprintf(fptr, "step %d : ", step);

  for(int i = 0; i < NUMSTATES; i++)
  {
    frac = (double)num_in_state[i]/ntot;
    fprintf(fptr, "Frac in state %d = %f,\t", i, frac);
  }

  fprintf(fptr, " \n");

  fclose(fptr);
}

// write timing data for gpu OpenACC code to file
int gpu_write_timing(struct Options const * opt, float elapsed_time,
                     float kernel_time)
{
  FILE *file = NULL;
  char filename[200];
  int ierr = 0;

  // create filename for given options
  sprintf(filename, "output/timing-gpu-openacc.n-%i.m-%i.nsteps-%i.txt",
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

int main(int argc, char **argv)
{
  // debug flags
  // - `debug_verbose != 0` will annotate, to stderr, the program as it is
  //   executed
  // - `debug_timing != 0` will annotate, to stderr, the timing variables as
  //   they are calculated
  // - `debug_visual != 0` will annotate, to stderr, the ascii visualisation of
  //   grid variables as they are initialised and updated
  const int debug_verbose = 0;
  const int debug_timing = 0;
  const int debug_visual = 0;

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

  // debug: verbose
  verbose("<debug_verbose> = on");
  if (debug_timing) verbose("<debug_timing> = on");
  if (debug_visual) verbose("<debug_visual> = on");

  // define timing variables
  struct timeval start;
  struct timeval gol_start;
  struct timeval step_start;
  struct timeval transfer_start;

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

  // allocate memory for `grid`, `update_grid` variables
  int *grid = (int *) malloc(sizeof(int) * n * m);
  int *updated_grid = (int *) malloc(sizeof(int) * n * m);

  if (grid == NULL || updated_grid == NULL)
  {
    fprintf(stderr, "error while allocating memory for grids\n");
    return -1;
  }

  verbose("<grid>, <updated_grid> memory allocated: sizeof(int) * %i", n * m);

  // initialise step counter, kernel_time
  int current_step = 0;
  float kernel_time = 0.0;

  // generate initial conditions
  generate_IC(opt->iictype, grid, n, m);

  verbose("<grid> initial conditions generated");

  // debug: visualise `grid` after initial conditions
  visual(current_step, grid, n, m, "<grid, initial> = ");

  // initialise timing of GOL simulation
  gol_start = init_time();

  verbose("GOL simulation timing initialised");

  // initialise timing of OpenACC data transfer in
  transfer_start = init_time();

  verbose("OpenACC: data transfer (in) timing initialised");

  // move `grid` to gpu, allocate `updated_grid` on gpu
#pragma acc enter data copyin(grid[0:n*m]) create(updated_grid[0:n*m])

  // debug: calculate time for OpenACC data transfer in
  float transfer_time = get_elapsed_time(transfer_start);
  timing("<transfer_time> = %f [ms]", transfer_time);

  verbose("OpenACC: <grid> copied to gpu, <updated_grid> allocated on gpu");

  // GOL simulation loop
  while (current_step != nsteps)
  {
    // increment step counter
    current_step++;

    verbose("<%i> GOL step started", current_step);

    // initialise timing of current step in GOL simulation
    step_start = init_time();

    verbose("<%i> timing initialised", current_step);

    // calculate next state of grid according to GOL update rules
#pragma acc parallel loop independent present(grid, updated_grid)
    for (int i = 0; i < n; i++)
    {
#pragma acc loop independent
      for (int j = 0; j < m; j++)
      {
        const int neighbours = gol_neighbours(grid, n, m, i, j);
        const int state = grid[i*m + j];
        updated_grid[i*m + j] = gol_update(state, neighbours);
      }
    }

    // calculate time spent in kernel
    kernel_time += get_elapsed_time(step_start);

    verbose("OpenACC: <%i> next GOL state calculated", current_step);

    // swap current and updated grid
    {
      int *tmp = grid;
      grid = updated_grid;
      updated_grid = tmp;
    }

    verbose("<%i> grids swapped", current_step);

    // debug: calculate time for this step in GOL simulation
    float step_time = get_elapsed_time(step_start);
    timing("<step_time, %i> = %f [ms]", current_step, step_time);

    // debug: visualise `grid` after current step
    if (debug_visual)
    {
      transfer_start = init_time();

      verbose("OpenACC: <%i> <grid> update timing initialised", current_step);

#pragma acc update self(grid[0:n*m])

      transfer_time = get_elapsed_time(transfer_start);
      timing("<transfer_time> = %f [ms]", transfer_time);

      verbose("OpenACC: <%i> updated <grid> on host", current_step);
    }
    visual(current_step, grid, n, m, "<grid, %i> = ", current_step);

    verbose("<%i> GOL step finished", current_step);
  }

  // initialise timing of OpenACC data transfer out
  transfer_start = init_time();

  verbose("OpenACC: data transfer (out) timing initialised");

  // move `grid` to cpu, delete `updated_grid` on gpu
#pragma acc exit data copyout(grid[0:n*m]) delete(updated_grid[0:n*m])

  // debug: calculate time for OpenACC data transfer out
  transfer_time = get_elapsed_time(transfer_start);
  timing("<transfer_time> = %f [ms]", transfer_time);

  verbose("OpenACC: <grid> copied from gpu, <updated_grid> deleted on gpu");

  // calculate time for GOL simulation
  float elapsed_time = get_elapsed_time(gol_start);
  timing("<elapsed_time> = %f [ms]", elapsed_time);

  // debug: total kernel time
  timing("<kernel_time> = %f [ms]", kernel_time);

  verbose("GOL simulation timing finished");

  // write timing to file
  gpu_write_timing(opt, elapsed_time, kernel_time);

  verbose("<elapsed_time>, <kernel_time> written to file");

  // debug: visualise `grid` after loop completion
  visual(current_step, grid, n, m, "<grid, final> = ");


  // write GOL statistics to file
  // game_of_life_stats(opt, current_step, grid);

  // free memory
  free(grid);
  free(updated_grid);
  free(opt);

  verbose("memory freed");

  // debug: calculate time for entire program execution
  float total_time = get_elapsed_time(start);
  timing("<total_time> = %f [ms]", total_time);

  verbose("program timing finished");

  return 0;
}
