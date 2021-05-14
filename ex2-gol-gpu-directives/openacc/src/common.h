/*!

*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <omp.h>
#ifdef USEPNG
#include <png.h>
#endif

#define bool char

/*! \name cell states
*/
//@{
enum CellState{ALIVE,DEAD,DYING,BORN};
#define NUMSTATES 4

//@}
/*! \name visualisation
*/
//@{
enum VisualiseType{VISUAL_ASCII,VISUAL_PNG,VISUAL_OPENGL,VISUAL_NONE};
//}

/*! \name IC choices
*/
//@{
enum ICType{IC_RAND, IC_FILE};
//#define ICRAND 0
//#define ICFILE 1
//@}

/*! \name Rules for GOL
*/
//@{
enum RuleType{RULE_STANDARD, RULE_EXTENDED, RULE_PROB};
//@}

/*! \name Neighbour choice
*/
//@{
enum NeighbourType{NEIGHBOUR_STANDARD, NEIGHBOUR_EXTENDED};
//@}

/*! \name Boundary type for GOL
*/
//@{
enum BoundaryType{BOUNDARY_HARD, BOUNDARY_TORAL, BOUNDARY_TORAL_X_HARD_Y, BOUNDARY_TORAL_Y_HARD_X};
//@}

/// store options for code
struct Options {
  int n, m, nsteps;
  enum ICType iictype;
  enum VisualiseType ivisualisetype;
  enum RuleType iruletype;
  enum NeighbourType ineighbourtype;
  enum BoundaryType iboundarytype;
  //store the file name to write stats too
  char statsfile[2000];
};

/// visualise the game of life
void visualise(enum VisualiseType ivisualisechoice, int step, int *grid, int n, int m);
/// ascii visualisation
void visualise_ascii(int step, int *grid, int n, int m);
/// png visualisation
void visualise_png(int step, int *grid, int n, int m);
/// ascii visualisation
void visualise_none(int step);

/// generate IC
void generate_IC(enum ICType ic_choice, int *grid, int n, int m);
/// generate random IC
void generate_rand_IC(int *grid, int n, int m);

/// get some basic timing info
struct timeval init_time();
/// get the elapsed time relative to start, return current wall time
struct timeval get_elapsed_time(struct timeval start);

/// GOL prototype
void game_of_life(struct Options *opt,
    int *current_grid, int *next_grid, int n, int m);

///GOL stats protoype
void game_of_life_stats(struct Options *opt, int steps,
    int *current_grid);

/// UI
void getinput(int argc, char **argv, struct Options *opt);

/*! PNG visualiation of GOL. Taken from https://www.lemoda.net/c/write-png/
*/
//@{
#ifdef USEPNG
/// A coloured pixel.
typedef struct
{
    uint8_t red;
    uint8_t green;
    uint8_t blue;
}
pixel_t;

/// A picture.
typedef struct
{
    pixel_t *pixels;
    size_t width;
    size_t height;
}
bitmap_t;

/*! Given "bitmap", this returns the pixel of bitmap at the point
   ("x", "y").
*/
static pixel_t * pixel_at (bitmap_t * bitmap, int x, int y)
{
    return bitmap->pixels + bitmap->width * y + x;
}

/*! Write "bitmap" to a PNG file specified by "path"; returns 0 on
   success, non-zero on error.
*/
static int save_png_to_file (bitmap_t *bitmap, const char *path)
{
    FILE * fp;
    png_structp png_ptr = NULL;
    png_infop info_ptr = NULL;
    size_t x, y;
    png_byte ** row_pointers = NULL;
    /* "status" contains the return value of this function. At first
       it is set to a value which means 'failure'. When the routine
       has finished its work, it is set to a value which means
       'success'. */
    int status = -1;
    /* The following number is set by trial and error only. I cannot
       see where it it is documented in the libpng manual.
    */
    int pixel_size = 3;
    int depth = 8;

    fp = fopen (path, "wb");
    if (! fp) {
        goto fopen_failed;
    }

    png_ptr = png_create_write_struct (PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (png_ptr == NULL) {
        goto png_create_write_struct_failed;
    }

    info_ptr = png_create_info_struct (png_ptr);
    if (info_ptr == NULL) {
        goto png_create_info_struct_failed;
    }

    /* Set up error handling. */

    if (setjmp (png_jmpbuf (png_ptr))) {
        goto png_failure;
    }

    /* Set image attributes. */

    png_set_IHDR (png_ptr,
                  info_ptr,
                  bitmap->width,
                  bitmap->height,
                  depth,
                  PNG_COLOR_TYPE_RGB,
                  PNG_INTERLACE_NONE,
                  PNG_COMPRESSION_TYPE_DEFAULT,
                  PNG_FILTER_TYPE_DEFAULT);

    /* Initialize rows of PNG. */

    row_pointers = png_malloc (png_ptr, bitmap->height * sizeof (png_byte *));
    for (y = 0; y < bitmap->height; y++) {
        png_byte *row =
            png_malloc (png_ptr, sizeof (uint8_t) * bitmap->width * pixel_size);
        row_pointers[y] = row;
        for (x = 0; x < bitmap->width; x++) {
            pixel_t * pixel = pixel_at (bitmap, x, y);
            *row++ = pixel->red;
            *row++ = pixel->green;
            *row++ = pixel->blue;
        }
    }

    /* Write the image data to "fp". */

    png_init_io (png_ptr, fp);
    png_set_rows (png_ptr, info_ptr, row_pointers);
    png_write_png (png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

    /* The routine has successfully written the file, so we set
       "status" to a value which indicates success. */

    status = 0;

    for (y = 0; y < bitmap->height; y++) {
        png_free (png_ptr, row_pointers[y]);
    }
    png_free (png_ptr, row_pointers);

 png_failure:
 png_create_info_struct_failed:
    png_destroy_write_struct (&png_ptr, &info_ptr);
 png_create_write_struct_failed:
    fclose (fp);
 fopen_failed:
    return status;
}
#endif
//@}
