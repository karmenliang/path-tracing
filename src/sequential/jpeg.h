#include <stdio.h>

#include "jpeglib.h"

/* 
 * The "frame structure" contains an image frame 
 * (in RGB or grayscale formats)
 */
typedef struct frame_struct
{
  JSAMPLE *image_buffer;        /* Points to large array of R,G,B-order/grayscale data
                                 * Access directly with:
                                 *   image_buffer[num_components*pixel + component]
                                 */

  JSAMPLE **row_pointers;       /* Points to an array of pointers to the beginning
                                 * of each row in the image buffer.  Use to access
                                 * the image buffer in a row-wise fashion, with:
                                 *   row_pointers[row][num_components*pixel + component]
                                 */

  int image_height;             /* Number of rows in image */
  int image_width;              /* Number of columns in image */
  int num_components;   /* Number of components (usually RGB=3 or gray=1) */
} frame_struct_t;

typedef frame_struct_t *frame_ptr;

frame_ptr output_frame;   /* Pointer to output frame */

/*
 * write_JPEG_file writes out the contents of an image buffer to a JPEG.
 * A quality level of 2-100 can be provided (default = 75, high quality = ~95,
 * low quality = ~25, utter pixellation = 2).  Note that unlike read_JPEG_file,
 * it does not do any memory allocation on the buffer passed to it.
 */

void write_JPEG_file (char * filename, frame_ptr p_info, int quality) {
    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr jerr;
    FILE * outfile;               /* target file */

    /* Step 1: allocate and initialize JPEG compression object */
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);

    /* Step 2: specify data destination (eg, a file) */
    /* Note: steps 2 and 3 can be done in either order. */
    if ((outfile = fopen(filename, "wb")) == NULL) {
        fprintf(stderr, "ERROR: Can't open output file %s\n", filename);
        exit(1);
    }
    jpeg_stdio_dest(&cinfo, outfile);

    /* Step 3: set parameters for compression */
    /* Set basic picture parameters (not optional) */
    cinfo.image_width = p_info->image_width;      /* image width and height, in pixels */
    cinfo.image_height = p_info->image_height;
    cinfo.input_components = p_info->num_components; /* # of color components per pixel */

    if (p_info->num_components == 3)
        cinfo.in_color_space = JCS_RGB;     /* colorspace of input image */
    else if (p_info->num_components == 1)
        cinfo.in_color_space = JCS_GRAYSCALE;
    else {
        fprintf(stderr, "ERROR: Non-standard colorspace for compressing!\n");
        exit(1);
    }

    /* Fill in the defaults for everything else, then override quality */
    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, quality, TRUE /* limit to baseline-JPEG values */);

    /* Step 4: Start compressor */
    jpeg_start_compress(&cinfo, TRUE);

    /* Step 5: while (scan lines remain to be written) */
    /*           jpeg_write_scanlines(...); */
    while (cinfo.next_scanline < cinfo.image_height) {
        (void) jpeg_write_scanlines(&cinfo, &(p_info->row_pointers[cinfo.next_scanline]), 1);
    }

    /* Step 6: Finish compression & close output */
    jpeg_finish_compress(&cinfo);
    fclose(outfile);

    /* Step 7: release JPEG compression object */
    jpeg_destroy_compress(&cinfo);
}

/*
 * allocate/destroy_frame allocates a frame_struct_t and fills in the
 * blanks appropriately (including allocating the actual frames), and
 * then destroy them afterwards.
 */

frame_ptr allocate_frame(int height, int width, int num_components) {

  int row_stride;		/* physical row width in output buffer */
  int i;
  frame_ptr p_info;		/* output frame information */

  /* JSAMPLEs per row in output buffer */
  row_stride = width * num_components;
  
  /* Basic struct and information */
  if ((p_info = (frame_struct_t*) malloc(sizeof(frame_struct_t))) == NULL) {
    fprintf(stderr, "ERROR: Memory allocation failure\n");
    exit(1);
  }
  p_info->image_height = height;
  p_info->image_width = width;
  p_info->num_components = num_components;

  /* Image array and pointers to rows */
  if ((p_info->row_pointers = (JSAMPLE**) malloc(sizeof(JSAMPLE *) * height)) == NULL) {
    fprintf(stderr, "ERROR: Memory allocation failure\n");
    exit(1);
  }
  if ((p_info->image_buffer = (JSAMPLE*) malloc(sizeof(JSAMPLE) * row_stride * height)) == NULL) {
    fprintf(stderr, "ERROR: Memory allocation failure\n");
    exit(1);
  }
  for (i=0; i < height; i++)
  	p_info->row_pointers[i] = & (p_info->image_buffer[i * row_stride]);
  
  /* And send it back! */
  return p_info;
}

void destroy_frame(frame_ptr kill_me)
{
	free(kill_me->image_buffer);
	free(kill_me->row_pointers);
	free(kill_me);
}
