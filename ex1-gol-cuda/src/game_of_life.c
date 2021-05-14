#include "common.h"

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
  Implements the game of life on a grid of size `n` times `m`, starting from the `initial_state` configuration.

  If `nsteps` is positive, returns the last state reached.
*/
int* cpu_game_of_life(const int *initial_state, int n, int m, int nsteps)
{
  int *grid = (int *) malloc(sizeof(int) * n * m);
  int *updated_grid = (int *) malloc(sizeof(int) * n * m);

  if (!grid || !updated_grid)
  {
    printf("Error while allocating memory.\n");
    exit(1);
  }

  int current_step = 0;
  int *tmp = NULL;

  memcpy(grid, initial_state, sizeof(int) * n * m);

  while(current_step != nsteps)
  {
    current_step++;

    // Uncomment the following line if you want to print the state at every step
    // visualise(opt->ivisualisetype, current_step, grid, n, m);

    cpu_game_of_life_step(grid, updated_grid, n, m);

    // swap current and updated grid
    tmp = grid;
    grid = updated_grid;
    updated_grid = tmp;
  }

  free(updated_grid);

  return grid;
}

// write timing data for cpu code to file
int cpu_write_timing(struct Options *opt, float elapsed_time)
{
  FILE *file = NULL;
  char filename[100];
  int ierr = 0;

  // create filename for given options
  sprintf(filename, "output/timing-cpu.n-%i.m-%i.nsteps-%i.txt", \
          opt->n, opt->m,opt->nsteps);

  printf("writing timing data to filename: %s\n", filename);

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
    fprintf(file, "%f\n", elapsed_time);

    // close file
    fclose(file);
  }

  return ierr;
}

#ifndef INCLUDE_CPU_VERSION // do not define the main function if this file is included somewhere else.
int main(int argc, char **argv)
{
  struct Options *opt = (struct Options *) malloc(sizeof(struct Options));

  getinput(argc, argv, opt);
  int n = opt->n, m = opt->m, nsteps = opt->nsteps;

  int *initial_state = (int *) malloc(sizeof(int) * n * m);

  if(!initial_state)
  {
    printf("Error while allocating memory.\n");
    return -1;
  }

  generate_IC(opt->iictype, initial_state, n, m);

  struct timeval start;
  start = init_time();

  int *final_state = cpu_game_of_life(initial_state, n, m, nsteps);

  float elapsed = get_elapsed_time(start);
  printf("Finished GOL in %f ms\n", elapsed);

  cpu_write_timing(opt, elapsed);

  free(final_state);
  free(initial_state);
  free(opt);
  return 0;
}
#endif
