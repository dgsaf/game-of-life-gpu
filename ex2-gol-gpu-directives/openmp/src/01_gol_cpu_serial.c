#include "common.h"

void game_of_life(struct Options *opt, int *current_grid, int *next_grid, \
                  int n, int m)
{
  int neighbours;
  int n_i[8], n_j[8];
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < m; j++)
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

      if (current_grid[i*m + j] == ALIVE \
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
  int m = opt->m, n = opt->n;

  for (int i = 0; i < NUMSTATES; i++) num_in_state[i] = 0;

  for (int j = 0; j < m; j++)
  {
    for (int i = 0; i < n; i++)
    {
      num_in_state[current_grid[i*m + j]]++;
    }
  }

  double frac, ntot = opt->m*opt->n;

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

  for (int i = 0; i < NUMSTATES; i++)
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

int main(int argc, char **argv)
{
  // read input parameters
  struct Options *opt = (struct Options *) malloc(sizeof(struct Options));
  getinput(argc, argv, opt);
  int n = opt->n, m = opt->m, nsteps = opt->nsteps;

  // generate initial conditions
  int *grid = (int *) malloc(sizeof(int) * n * m);
  int *updated_grid = (int *) malloc(sizeof(int) * n * m);

  if (!grid || !updated_grid)
  {
    printf("Error while allocating memory.\n");
    return -1;
  }

  int current_step = 0;
  int *tmp = NULL;

  generate_IC(opt->iictype, grid, n, m);

  // initialise timing
  struct timeval start, step_start;
  float step_time;
  start = init_time();

  // calculate final game_of_life state
  while (current_step != nsteps)
  {
    step_start = init_time();

    game_of_life(opt, grid, updated_grid, n, m);

    // swap current and updated grid
    tmp = grid;
    grid = updated_grid;
    updated_grid = tmp;

    current_step++;

    step_time = get_elapsed_time(step_start);
  }

  // finalise timing and write output
  float elapsed_time = get_elapsed_time(start);
  printf("Finished GOL in %f ms\n", elapsed_time);

  cpu_write_timing(opt, elapsed_time);

  // game_of_life_stats(opt, current_step, grid);
  visualise(VISUAL_ASCII, current_step, grid, n, m);

  // free memory
  free(grid);
  free(updated_grid);
  free(opt);

  return 0;
}
