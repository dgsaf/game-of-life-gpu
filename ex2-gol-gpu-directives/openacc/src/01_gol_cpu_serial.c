#include "common.h"

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
    fprintf(stderr, "01_gol_cpu_serial.c: cannot open filename: %s\n", filename);
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

int main(int argc, char **argv)
{
  // debug parameters
  const int debug_verbose = 1;
  const int debug_timing = 1;
  const int debug_visual = 1;

  const int visual_n_max = 20;
  const int visual_m_max = 20;

#define min(x, y) (((x) < (y)) ? (x) : (y))
#define debug_visualise(step, grid, n, m) {                 \
    visualise(VISUAL_ASCII, step, grid,                     \
              min(visual_n_max, n), min(visual_m_max, m));  \
  }

  // define timing variables
  struct timeval start;
  struct timeval gol_start;
  struct timeval step_start;

  // initialise timing of entire program execution
  start = init_time();

  // read input
  struct Options *opt = (struct Options *) malloc(sizeof(struct Options));
  getinput(argc, argv, opt);

  // define parameter variables
  const int n = opt->n;
  const int m = opt->m;
  const int nsteps = opt->nsteps;

  // allocate memory for `grid`, `update_grid` variables
  int *grid = (int *) malloc(sizeof(int) * n * m);
  int *updated_grid = (int *) malloc(sizeof(int) * n * m);

  if (grid == NULL || updated_grid == NULL)
  {
    printf("01_gol_cpu_serial.c: error while allocating memory for grids\n");
    return -1;
  }

  // initialise step counter
  int current_step = 0;

  // generate initial conditions
  generate_IC(opt->iictype, grid, n, m);

  // debug: visualise `grid` after initial conditions
  if (debug_visual)
  {
    printf("<grid, initial> = \n", current_step);
    debug_visualise(current_step, grid, n, m);
  }

  // initialise timing of GOL simulation
  gol_start = init_time();

  // GOL simulation loop
  while (current_step != nsteps)
  {
    // initialise timing of current step in GOL simulation
    step_start = init_time();

    // calculate next state of grid according to GOL update rules
    game_of_life(opt, grid, updated_grid, n, m);

    // swap current and updated grid
    {
      int *tmp = grid;
      grid = updated_grid;
      updated_grid = tmp;
    }

    // debug: calculate time for this step in GOL simulation
    if (debug_timing)
    {
      float step_time = get_elapsed_time(step_start);
      printf("<step_time, %i> = %f [ms]\n", current_step, step_time);
    }

    // debug: visualise `grid` after current step
    if (debug_visual)
    {
      printf("<grid, %i> = \n", current_step);
      debug_visualise(current_step, grid, n, m);
    }

    // increment step counter
    current_step++;
  }

  // calculate time for GOL simulation
  float elapsed_time = get_elapsed_time(gol_start);
  printf("<elapsed_time> = %f [ms]\n", elapsed_time);

  // debug: calculate time for entire program execution
  if (debug_timing)
  {
    float total_time = get_elapsed_time(start);
    printf("<total_time> = %f [ms]\n", total_time);
  }

  // debug: visualise `grid` after loop completion
  if (debug_visual)
  {
    printf("<grid, final> = \n", current_step);
    debug_visualise(current_step, grid, n, m);
  }

  // write total time to file
  cpu_write_timing(opt, elapsed_time);

  // write GOL statistics to file
  // game_of_life_stats(opt, current_step, grid);

  // free memory
  free(grid);
  free(updated_grid);
  free(opt);

  return 0;
}
