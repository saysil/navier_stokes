
#ifndef FLUID_CL_H
#define FLUID_CL_H

#include <stdio.h>

#include <CL/cl.h>

#include "fluid_core.h"

struct fluid_cl_info { 
    cl_context       context;
    cl_command_queue command_queue;
    cl_program       stokes_program;
    cl_kernel        diffusev_computek;
    cl_kernel        advectv_computek;
    cl_kernel        divergence_computek;
    cl_kernel        pressure_computek;
    cl_kernel        pressure_gradient_computek;
    cl_kernel        velocity_computek;
    cl_kernel        diffuse_computek;

    cl_program       color_program;
    cl_kernel        hue_computek;
    cl_kernel        pixels_computek;
};

struct fluid_cl_info 
fluid_cl_init(void);

void 
fluid_compute_velocity(cl_float2    *velocity,
                       char         *impossible_list,
                       int           width,
                       int           height,
                       float         tile_distance,
                       float         time_interval,
                       float         viscosity,
                struct fluid_cl_info info);

void 
fluid_compute_concentration(float        *concentration,
                            cl_float2    *velocity,
                            char         *impossible_list,
                            int           width,
                            int           height,
                            float         d_coeff,
                            float         time_interval,
                            float         tile_distance,
                     struct fluid_cl_info info);


void
fluid_compute_pixels(int          *pixels,
                     float        *concentration,
                     int           width,
                     int           height,
                     float         scale,
              struct fluid_cl_info info);

#define cl_float2_padded_screen_size (sizeof(cl_float2)*((height+2)*(width+2)))


#define fluid_clSetKernelArg(_kernel, _index, _arg_size, _arg_value, ret) \
if ((ret = clSetKernelArg(_kernel, _index, _arg_size, _arg_value)) != CL_SUCCESS) {\
    fprintf(stderr, "Could not set kernel argument #%d, " #_arg_value " in kernel " #_kernel ". Error code: %d\n", _index, ret);\
}


#endif

