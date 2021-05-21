#include "common.h"
#include <omp.h>

//
#pragma omp declare target
int game_of_life_neighbours(int *current_grid, int n, int m, int i, int j)
{
  // index the neighbourhood clockwise around the current cell
  int n_i[8], n_j[8];

  n_i[0] = i - 1; n_j[0] = j - 1;
  n_i[1] = i - 1; n_j[1] = j;
  n_i[2] = i - 1; n_j[2] = j + 1;
  n_i[3] = i;     n_j[3] = j + 1;
  n_i[4] = i + 1; n_j[4] = j + 1;
  n_i[5] = i + 1; n_j[5] = j;
  n_i[6] = i + 1; n_j[6] = j - 1;
  n_i[7] = i;     n_j[7] = j - 1;

  // count the number of living neighbours
  int neighbours = 0;

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

  return neighbours;
}
#pragma omp end declare target

//
#pragma omp declare target
int game_of_life_next_state(int current_state, int neighbours)
{
  int next_state = DEAD;

  if (current_state == ALIVE && (neighbours == 2 || neighbours == 3))
  {
    next_state = ALIVE;
  }
  else if (current_state == DEAD && neighbours == 3)
  {
    next_state = ALIVE;
  }
  else
  {
    next_state = DEAD;
  }

  return next_state;
}
#pragma omp end declare target

// game_of_life
// This function has been simplified from its original form in the following
// ways:
// - An atomic function has been introduced for counting the number of living
//   neighbours, `game_of_life_neighbours()`.
// - An atomic function has been introduced for determining the next state of a
//   cell, based on its current state and the number of living neighbours,
//   `game_of_life_next_state()`.
#pragma omp declare target
void game_of_life(struct Options *opt, int *current_grid, int *next_grid, \
                  int n, int m)
{
  for (int i = 0; i < n; i++)
  {
    for (int j = 0; j < m; j++)
    {
      int n_i[8], n_j[8];
      int neighbours;

      // index the neighbourhood around current cell
      n_i[0] = i - 1; n_j[0] = j - 1;
      n_i[1] = i - 1; n_j[1] = j;
      n_i[2] = i - 1; n_j[2] = j + 1;
      n_i[3] = i;     n_j[3] = j + 1;
      n_i[4] = i + 1; n_j[4] = j + 1;
      n_i[5] = i + 1; n_j[5] = j;
      n_i[6] = i + 1; n_j[6] = j - 1;
      n_i[7] = i;     n_j[7] = j - 1;

      // count the number of living neighbours
      neighbours = 0;

      if (n_i[0] >= 0 && n_j[0] >= 0                            \
          && current_grid[n_i[0] * m + n_j[0]] == ALIVE) neighbours++;
      if (n_i[1] >= 0                                           \
          && current_grid[n_i[1] * m + n_j[1]] == ALIVE) neighbours++;
      if (n_i[2] >= 0 && n_j[2] < m                             \
          && current_grid[n_i[2] * m + n_j[2]] == ALIVE) neighbours++;
      if (n_j[3] < m                                            \
          && current_grid[n_i[3] * m + n_j[3]] == ALIVE) neighbours++;
      if (n_i[4] < n && n_j[4] < m                              \
          && current_grid[n_i[4] * m + n_j[4]] == ALIVE) neighbours++;
      if (n_i[5] < n                                            \
          && current_grid[n_i[5] * m + n_j[5]] == ALIVE) neighbours++;
      if (n_i[6] < n && n_j[6] >= 0                             \
          && current_grid[n_i[6] * m + n_j[6]] == ALIVE) neighbours++;
      if (n_j[7] >= 0                                           \
          && current_grid[n_i[7] * m + n_j[7]] == ALIVE) neighbours++;

      // update state
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
}
#pragma omp end declare target

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

// write timing data for gpu code to file
int gpu_write_timing(struct Options const * opt, float const elapsed_time, \
                     float const kernel_time)
{
  FILE *file = NULL;
  char filename[200];
  int ierr = 0;

  // create filename for given options
  sprintf(filename, "output/timing-gpu-openmp.n-%i.m-%i.nsteps-%i.txt", \
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

  generate_IC(opt->iictype, grid, n, m);

  // initialise timing
  struct timeval start, kernel_start;
  start = init_time();
  float kernel_time = 0.0;

  // omp - move grid variables from cpu memory to gpu memory for the duration of
  // the loop
  // - `target` defines a task to perform on the gpu
  // - `enter data` defines a transfer of cpu memory to gpu memory
  // - `map(to:*)` moves `grid` and `updated_grid` from the cpu memory into the
  //   gpu memory
#pragma omp target enter data map(to:grid[0:n*m],updated_grid[0:n*m])

  // calculate final game_of_life state
  while (current_step != nsteps)
  {
#pragma omp single
    kernel_start = init_time();

    // perform game_of_life step
#pragma omp target
#pragma omp teams
#pragma omp distribute parallel for collapse(2) schedule(static)
    for (int i = 0; i < n; i++)
    {
      for (int j = 0; j < m; j++)
      {
        int neighbours = game_of_life_neighbours(grid, n, m, i, j);

        updated_grid[i*m + j] = game_of_life_next_state(grid[i*m + j], \
                                                        neighbours);
      }
    }

    // wait for all cells to be updated
#pragma omp barrier

#pragma omp single
    kernel_time += get_elapsed_time(kernel_start);

    // swap current and updated grid
#pragma omp single
    {
      int *tmp = NULL;
      tmp = grid;
      grid = updated_grid;
      updated_grid = tmp;
    }

    // wait before re-entering loop
#pragma omp barrier

#pragma omp single
    current_step++;
  }

  // omp - retrieve grid variables from gpu memory to cpu memory
  // - `target` defines a task to perform on the gpu
  // - `exit data` defines a transfer of gpu memory to cpu memory
  // - `map(from:*)` moves `grid` and `updated_grid` from the gpu memory into
  //   the cpu memory
#pragma omp target exit data map(from:grid[0:n*m],updated_grid[0:n*m])

  // finalise timing and write output
  float elapsed_time = get_elapsed_time(start);
  printf("Finished GOL in %f ms\n", elapsed_time);

  gpu_write_timing(opt, elapsed_time, kernel_time);

  // game_of_life_stats(opt, current_step, grid);
  visualise(VISUAL_ASCII, current_step, grid, n, m);

  // free memory
  free(grid);
  free(updated_grid);
  free(opt);

  return 0;
}
