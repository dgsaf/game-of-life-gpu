#include "common.h"

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

void cpu_game_of_life_step(int *current_grid, int *next_grid, int n, int m)
{
  int neighbours;
  int n_i[8], n_j[8];
  for(int i = 0; i < n; i++)
  {
    for(int j = 0; j < m; j++)
    {
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

      if (n_i[0] >= 0 && n_j[0] >= 0 \
          && current_grid[n_i[0] * m + n_j[0]] == ALIVE) neighbours++;
      if (n_i[1] >= 0                                                   \
          && current_grid[n_i[1] * m + n_j[1]] == ALIVE) neighbours++;
      if (n_i[2] >= 0 && n_j[2] < m \
          && current_grid[n_i[2] * m + n_j[2]] == ALIVE) neighbours++;
      if (n_j[3] < m \
          && current_grid[n_i[3] * m + n_j[3]] == ALIVE) neighbours++;
      if (n_i[4] < n && n_j[4] < m \
          && current_grid[n_i[4] * m + n_j[4]] == ALIVE) neighbours++;
      if (n_i[5] < n \
          && current_grid[n_i[5] * m + n_j[5]] == ALIVE) neighbours++;
      if (n_i[6] < n && n_j[6] >= 0 \
          && current_grid[n_i[6] * m + n_j[6]] == ALIVE) neighbours++;
      if (n_j[7] >= 0 \
          && current_grid[n_i[7] * m + n_j[7]] == ALIVE) neighbours++;

      if (current_grid[i*m + j] == ALIVE  \
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


/*
  Implements the game of life on a grid of size `n` times `m`, starting from
  the `initial_state` configuration.

  If `nsteps` is positive, returns the last state reached.
*/
int* cpu_game_of_life(const int *initial_state, int n, int m, int nsteps)
{
  // allocate memory for `grid`, `update_grid` variables
  int *grid = (int *) malloc(sizeof(int) * n * m);
  int *updated_grid = (int *) malloc(sizeof(int) * n * m);

  if (grid == NULL || updated_grid == NULL)
  {
    fprintf(stderr, "error while allocating memory for grids\n");
    exit(1);
  }

  verbose("<grid>, <updated_grid> memory allocated: sizeof(int) * %i", n * m);

  // initialise step counter
  int current_step = 0;

  // copy `initial_state` to `grid`
  memcpy(grid, initial_state, sizeof(int) * n * m);

  verbose("copied <initial_state> to <grid>");

  // define timing variable
  struct timeval step_start;

  while (current_step != nsteps)
  {
    current_step++;

    verbose("<%i> GOL step started", current_step);

    // initialise timing of current step in GOL simulation
    step_start = init_time();

    verbose("<%i> timing initialised", current_step);

    // Uncomment the following line if you want to print the state at every step
    // visualise(opt->ivisualisetype, current_step, grid, n, m);

    // calculate next state of grid according to GOL update rules
    cpu_game_of_life_step(grid, updated_grid, n, m);

    verbose("<%i> next GOL state calculated", current_step);

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
    visual(current_step, grid, n, m, "<grid, %i> = ", current_step);

    verbose("<%i> GOL step finished", current_step);
  }

  verbose("GOL loop finished");

  free(updated_grid);

  verbose("<updated_grid> memory freed");

  return grid;
}

// write timing data for cpu code to file
int cpu_write_timing(struct Options const * opt, float const elapsed_time)
{
  FILE *file = NULL;
  char filename[200];
  int ierr = 0;

  // create filename for given options
  sprintf(filename, "output/timing-cpu.n-%i.m-%i.nsteps-%i.txt", \
          opt->n, opt->m, opt->nsteps);

  printf("writing cpu timing data to filename: %s\n", filename);

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
    fprintf(file, "# cpu_elapsed_time\n");
    fprintf(file, "# [ms]\n");
    fprintf(file, "%f\n", elapsed_time);

    // close file
    fclose(file);
  }

  return ierr;
}

// do not define the main function if this file is included somewhere else.
#ifndef INCLUDE_CPU_VERSION
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

  // calculate `final_state`
  int *final_state = cpu_game_of_life(initial_state, n, m, nsteps);

  // calculate time for GOL simulation
  float elapsed_time = get_elapsed_time(gol_start);
  timing("<elapsed_time> = %f [ms]", elapsed_time);

  verbose("GOL simulation timing finished");

  // write timing to file
  cpu_write_timing(opt, elapsed_time);

  verbose("<elapsed_time> written to file");

  // debug: visualise `final_state` after loop completion
  visual(nsteps, final_state, n, m, "<final_state> = ");

  // free memory
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
