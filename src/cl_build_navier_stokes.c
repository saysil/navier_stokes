//TEMPORARY OPENCL TESTING


#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>

#include <errno.h>

#include "fluid_cl.h"
#include "fluid_core.h"

struct fluid_cl_info 
fluid_cl_init(void)
{
    FILE  *stokes_fp, *diffusion_fp;
    size_t source_sizes[2]; // sizes of both source programs
    char  *source_strings[2]; //source strings of both programs

    FILE *color_fp;
    size_t color_size;
    char *color_string;


    stokes_fp = fopen("src/navier_stokes.cl", "r");
    diffusion_fp = fopen("src/diffusion.cl", "r");


    color_fp = fopen("src/concentration_to_color.cl", "r");


    // TODO: Do this properly
    fseek(stokes_fp, 0, SEEK_END);
    fseek(diffusion_fp, 0, SEEK_END);

    fseek(color_fp, 0, SEEK_END);

    source_sizes[0] = ftell(stokes_fp);
    source_sizes[1] = ftell(diffusion_fp);

    color_size = ftell(color_fp);

    rewind(stokes_fp);
    rewind(diffusion_fp);

    rewind(color_fp);

    printf("Source sizes acquired:\nStokes: %d\nDiffusion:%d\n", (int)source_sizes[0], (int)source_sizes[1]);

    source_strings[0] = malloc(source_sizes[0]);
    source_strings[1] = malloc(source_sizes[1]);

    color_string = malloc(color_size);

    fread(source_strings[0], 1, source_sizes[0], stokes_fp);
    fread(source_strings[1], 1, source_sizes[1], diffusion_fp);

    fread(color_string, 1, color_size, color_fp);

    fclose(stokes_fp);
    fclose(diffusion_fp);

    fclose(color_fp);

    printf("Loaded kernel strings.\n");

    struct fluid_cl_info info;
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    cl_int ret;
    cl_program stokes_program, color_program;
    cl_uint ret_num_devices;
    cl_uint ret_num_platforms;
    cl_context context = NULL;
    cl_command_queue command_queue = NULL;

    cl_kernel diffusev_compute, advectv_compute, 
              divergence_compute, pressure_compute, 
              pressure_gradient_compute, velocity_compute, 
              diffuse_compute = NULL;

    cl_kernel hue_compute, pixels_compute = NULL;

    ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);
    ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, &ret_num_devices);

    context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &ret);

    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to create CL Context.\n");
    }

    stokes_program = clCreateProgramWithSource(context, 2, (const char **)source_strings, source_sizes, &ret);

    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to create stokes program object. %d\n", ret);
    }

    color_program = clCreateProgramWithSource(context, 1, (const char **)&color_string, &color_size, &ret);

    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to create color program object. %d\n", ret);
    }
 
    ret = clBuildProgram(stokes_program, 1, &device_id, "-cl-std=CL2.0 -I ./inc/", NULL, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to stokes build CL Kernel.\n");

        printf("%d\n", ret);

        size_t log_size;
        clGetProgramBuildInfo(stokes_program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        char *cl_log = malloc(log_size);

        clGetProgramBuildInfo(stokes_program, device_id, CL_PROGRAM_BUILD_LOG, log_size, cl_log, NULL);

        printf("%s\n", cl_log);

        free(cl_log);

    } else {
        fprintf(stderr, "Successfully built program.\n");
    }


    ret = clBuildProgram(color_program, 1, &device_id, "-cl-std=CL2.0 -I ./inc/", NULL, NULL);
    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Failed to stokes build CL Kernel.\n");

        printf("%d\n", ret);

        size_t log_size;
        clGetProgramBuildInfo(color_program, device_id, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

        char *cl_log = malloc(log_size);

        clGetProgramBuildInfo(color_program, device_id, CL_PROGRAM_BUILD_LOG, log_size, cl_log, NULL);

        printf("%s\n", cl_log);

        free(cl_log);

    } else {
        fprintf(stderr, "Successfully built program.\n");
    }


    command_queue = clCreateCommandQueue(context, device_id, 0, &ret);

    diffusev_compute = clCreateKernel(stokes_program, "fluid_diffuse_velocity", &ret);
    advectv_compute = clCreateKernel(stokes_program, "fluid_convect_velocity", &ret);
    divergence_compute = clCreateKernel(stokes_program, "fluid_calculate_divergence", &ret);
    pressure_compute = clCreateKernel(stokes_program, "fluid_pressure_field", &ret);
    pressure_gradient_compute = clCreateKernel(stokes_program, "fluid_pressure_gradient", &ret);
    velocity_compute = clCreateKernel(stokes_program, "fluid_velocity_calculate", &ret);
    diffuse_compute = clCreateKernel(stokes_program, "fluid_diffuse_compute", &ret);
    printf("%d\n", ret);

    hue_compute = clCreateKernel(color_program, "concentration_to_hue", &ret);
    pixels_compute = clCreateKernel(color_program, "hue_to_rgb", &ret);

    ret = clFlush(command_queue);
    free(source_strings[0]);
    free(source_strings[1]);

    free(color_string);

    info.context = context;
    info.command_queue = command_queue;
    info.stokes_program = stokes_program;
    info.diffusev_computek = diffusev_compute;
    info.advectv_computek = advectv_compute;
    info.divergence_computek = divergence_compute;
    info.pressure_computek = pressure_compute;
    info.pressure_gradient_computek = pressure_gradient_compute;
    info.velocity_computek = velocity_compute;
    info.diffuse_computek = diffuse_compute;

    info.color_program = color_program;
    info.hue_computek = hue_compute;
    info.pixels_computek = pixels_compute;

    return info;


}

