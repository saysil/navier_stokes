
#include <stdio.h>
#include <stdlib.h>

#include <CL/cl.h>

#include "fluid_cl.h"
#include "fluid_core.h"


void
fluid_diffuse_velocity(cl_mem        velocity_obj,
                       cl_mem        impossible_obj,
                       int           width,
                       int           height,
                       float         viscosity,
                       float         tile_distance,
                       float         time_interval,
                       size_t       *work_offset,
                       size_t       *global_work_size,
                       size_t       *local_work_size,
                struct fluid_cl_info info)
{
    int ret = 0;
    
    cl_mem velocity1_obj, velocity2_obj;

    velocity1_obj = clCreateBuffer(info.context, CL_MEM_READ_WRITE, cl_float2_padded_screen_size, NULL, &ret);
    velocity2_obj = clCreateBuffer(info.context, CL_MEM_READ_WRITE, cl_float2_padded_screen_size, NULL, &ret);

    
    fluid_clSetKernelArg(info.diffusev_computek, 0, sizeof(cl_mem), (void *)&velocity_obj, ret);
    fluid_clSetKernelArg(info.diffusev_computek, 1, sizeof(cl_mem), (void *)&velocity1_obj, ret);
    fluid_clSetKernelArg(info.diffusev_computek, 2, sizeof(cl_mem), (void *)&velocity2_obj, ret);
    fluid_clSetKernelArg(info.diffusev_computek, 3, sizeof(cl_mem), (void *)&impossible_obj, ret);
    fluid_clSetKernelArg(info.diffusev_computek, 4, sizeof(int),    (void *)&width, ret);
    fluid_clSetKernelArg(info.diffusev_computek, 5, sizeof(int),    (void *)&height, ret);
    fluid_clSetKernelArg(info.diffusev_computek, 6, sizeof(float),  (void *)&viscosity, ret);
    fluid_clSetKernelArg(info.diffusev_computek, 7, sizeof(float),  (void *)&tile_distance, ret);
    fluid_clSetKernelArg(info.diffusev_computek, 8, sizeof(float),  (void *)&time_interval, ret);

    fluid_clSetKernelArg(info.advectv_computek, 2, sizeof(int),   (void *)&width, ret);
    fluid_clSetKernelArg(info.advectv_computek, 3, sizeof(int),   (void *)&height, ret);
    fluid_clSetKernelArg(info.advectv_computek, 4, sizeof(float), (void *)&tile_distance, ret);
    fluid_clSetKernelArg(info.advectv_computek, 5, sizeof(float), (void *)&time_interval, ret);

    cl_event init_complete[2];

    char initval = 0;

    //Copy velocity into velocity1 and zero velocity2.
    ret = clEnqueueCopyBuffer(info.command_queue, velocity_obj, velocity1_obj, 0, 0, cl_float2_padded_screen_size, 0, NULL, &init_complete[0]);
    ret = clEnqueueFillBuffer(info.command_queue, velocity2_obj, &initval, 1, 0, cl_float2_padded_screen_size, 0, NULL, &init_complete[1]);

    clWaitForEvents(1, init_complete);

    clReleaseEvent(init_complete[0]);
    clReleaseEvent(init_complete[1]);

    cl_event diffuse_complete;
    for (int i=0; i<20; i++) {
        ret = clEnqueueNDRangeKernel(info.command_queue, info.diffusev_computek, 2,
                                     work_offset, global_work_size, local_work_size,
                                     0, NULL, &diffuse_complete);

        if (ret != CL_SUCCESS) {
            printf("Error queueing kernel, %d\n", ret);
        }

        clWaitForEvents(1, &diffuse_complete);

        clReleaseEvent(diffuse_complete);

        cl_mem tmp = velocity1_obj;
        velocity1_obj = velocity2_obj;
        velocity2_obj = tmp;

        fluid_clSetKernelArg(info.diffusev_computek, 1, sizeof(cl_mem), &velocity1_obj, ret);
        fluid_clSetKernelArg(info.diffusev_computek, 2, sizeof(cl_mem), &velocity2_obj, ret);

    }

    fluid_clSetKernelArg(info.advectv_computek, 0, sizeof(cl_mem), &velocity1_obj, ret);
    fluid_clSetKernelArg(info.advectv_computek, 1, sizeof(cl_mem), &velocity2_obj, ret);

    cl_event advect_complete;
    ret = clEnqueueNDRangeKernel(info.command_queue, info.advectv_computek, 2,
                                 work_offset, global_work_size, local_work_size,
                                 0, NULL, &advect_complete);

    if (ret != CL_SUCCESS) {
        printf("Error copying buffer advect, %d\n", ret);
    }

    cl_event vcopy_complete;

    ret = clEnqueueCopyBuffer(info.command_queue, velocity2_obj, velocity_obj, 0, 0, cl_float2_padded_screen_size, 1, &advect_complete, &vcopy_complete);

    if (ret != CL_SUCCESS) {
        printf("Error copying buffer, %d\n", ret);
    }

    clWaitForEvents(1, &vcopy_complete);

    clReleaseMemObject(velocity1_obj);
    clReleaseMemObject(velocity2_obj);

}


void 
fluid_compute_velocity(cl_float2    *velocity,
                       char         *impossible_list,
                       int           width,
                       int           height,
                       float         tile_distance,
                       float         time_interval,
                       float         viscosity,
                struct fluid_cl_info info)
{

    cl_mem velocity_obj, velocity1_obj, velocity2_obj, impossible_obj, pressure_obj, pressure2_obj, pressure_gradient_obj, divergence_obj;

    cl_int ret;


    cl_event pressure_complete, velocity_complete;


    cl_event velocity_trigger[2];


    velocity_obj   = clCreateBuffer(info.context, CL_MEM_READ_WRITE, cl_float2_padded_screen_size, NULL, &ret);
    impossible_obj = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(char)*((height+2)*(width+2)), NULL, &ret);
    pressure_obj   = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(float)*((height+2)*(width+2)), NULL, &ret);
    pressure2_obj  = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(float)*((height+2)*(width+2)), NULL, &ret); 
    divergence_obj = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(float)*height*width, NULL, &ret);
    pressure_gradient_obj = clCreateBuffer(info.context, CL_MEM_READ_WRITE, cl_float2_padded_screen_size, NULL, &ret);

    fluid_clSetKernelArg(info.divergence_computek, 0, sizeof(cl_mem), (void *)&velocity_obj, ret);
    fluid_clSetKernelArg(info.divergence_computek, 1, sizeof(cl_mem), (void *)&divergence_obj, ret);
    fluid_clSetKernelArg(info.divergence_computek, 2, sizeof(cl_int), (void *)&width, ret);
    fluid_clSetKernelArg(info.divergence_computek, 3, sizeof(cl_int), (void *)&height, ret);
    fluid_clSetKernelArg(info.divergence_computek, 4, sizeof(float),  (void *)&tile_distance, ret);
 
    fluid_clSetKernelArg(info.pressure_computek, 0, sizeof(cl_mem), (void *)&velocity_obj, ret);
    fluid_clSetKernelArg(info.pressure_computek, 1, sizeof(cl_mem), (void *)&pressure_obj, ret);
    fluid_clSetKernelArg(info.pressure_computek, 2, sizeof(cl_mem), (void *)&pressure2_obj, ret);
    fluid_clSetKernelArg(info.pressure_computek, 3, sizeof(cl_mem), (void *)&divergence_obj, ret);
    fluid_clSetKernelArg(info.pressure_computek, 4, sizeof(cl_int), (void *)&width, ret);
    fluid_clSetKernelArg(info.pressure_computek, 5, sizeof(cl_int), (void *)&height, ret);
    fluid_clSetKernelArg(info.pressure_computek, 6, sizeof(float),  (void *)&tile_distance, ret);
    fluid_clSetKernelArg(info.pressure_computek, 7, sizeof(float),  (void *)&time_interval, ret);
   
    fluid_clSetKernelArg(info.pressure_gradient_computek, 0, sizeof(cl_mem), (void *)&pressure_obj, ret);
    fluid_clSetKernelArg(info.pressure_gradient_computek, 1, sizeof(cl_mem), (void *)&pressure_gradient_obj, ret);
    fluid_clSetKernelArg(info.pressure_gradient_computek, 2, sizeof(cl_int), (void *)&width, ret);
    fluid_clSetKernelArg(info.pressure_gradient_computek, 3, sizeof(cl_int), (void *)&height, ret);
    fluid_clSetKernelArg(info.pressure_gradient_computek, 4, sizeof(float),  (void *)&tile_distance, ret);

    fluid_clSetKernelArg(info.velocity_computek, 0, sizeof(cl_mem), (void *)&velocity_obj, ret);
    fluid_clSetKernelArg(info.velocity_computek, 1, sizeof(cl_mem), (void *)&impossible_obj, ret);
    fluid_clSetKernelArg(info.velocity_computek, 2, sizeof(cl_mem), (void *)&pressure_gradient_obj, ret);
    fluid_clSetKernelArg(info.velocity_computek, 3, sizeof(float),  (void *)&viscosity, ret);
    fluid_clSetKernelArg(info.velocity_computek, 4, sizeof(cl_int), (void *)&width, ret);
    fluid_clSetKernelArg(info.velocity_computek, 5, sizeof(cl_int), (void *)&height, ret);
    fluid_clSetKernelArg(info.velocity_computek, 6, sizeof(float),  (void *)&time_interval, ret);

    cl_event writes_complete[2];

    ret = clEnqueueWriteBuffer(info.command_queue, velocity_obj, CL_TRUE, 0, cl_float2_padded_screen_size, velocity, 0, NULL, &writes_complete[0]);
    ret = clEnqueueWriteBuffer(info.command_queue, impossible_obj, CL_TRUE, 0, sizeof(char)*((height+2)*(width+2)), impossible_list, 0, NULL, &writes_complete[1]);

    clWaitForEvents(2, writes_complete);

    size_t work_offset[2] = {0, 0};
    size_t global_work_size[2] = {width, height};
    size_t local_work_size[2] = {8, 8};


    fluid_diffuse_velocity(velocity_obj, impossible_obj, width, height, viscosity, tile_distance, time_interval, 
                           work_offset, global_work_size, local_work_size, info);

    cl_event init_done[2];

    char initval = 0;
    ret = clEnqueueFillBuffer(info.command_queue, pressure_obj, &initval, 1, 0, (width+2)*(height+2)*sizeof(float), 0, NULL, &init_done[0]);
    ret = clEnqueueFillBuffer(info.command_queue, pressure2_obj, &initval, 1, 0, (width+2)*(height+2)*sizeof(float), 0, NULL, &init_done[1]);

    clWaitForEvents(2, init_done);

    cl_event divergence_complete;
    clEnqueueNDRangeKernel(info.command_queue, info.divergence_computek, 2,
                           work_offset, global_work_size, local_work_size,
                           0, NULL, &divergence_complete);

    clWaitForEvents(1, &divergence_complete);

    /* Convergence after about 30 iterations */
    for (int i=0; i<30; i++) {
        ret = clEnqueueNDRangeKernel(info.command_queue, info.pressure_computek, 2,
                                     work_offset, global_work_size, local_work_size,
                                     0, NULL, &pressure_complete);

        if (ret !=CL_SUCCESS) {
            printf("Error: %d\n", ret);
        }


        clWaitForEvents(1, &pressure_complete);

        clReleaseEvent(pressure_complete);
        cl_mem tmp;
        tmp = pressure_obj;
        pressure_obj = pressure2_obj;
        pressure2_obj = tmp;
        fluid_clSetKernelArg(info.pressure_computek, 1, sizeof(cl_mem), (void *)&pressure_obj, ret);
        fluid_clSetKernelArg(info.pressure_computek, 2, sizeof(cl_mem), (void *)&pressure2_obj, ret);

    }

/*
    ret = clEnqueueNDRangeKernel(info.command_queue, info.pressure_computek, 2,
                                 work_offset, global_work_size, local_work_size,
                                 0, NULL, &pressure_complete);
    
    if (ret != CL_SUCCESS) {
        printf("Error executing kernel: %d\n", ret);
    } 
*/

    ret = clEnqueueNDRangeKernel(info.command_queue, info.pressure_gradient_computek, 2,
                                 work_offset, global_work_size, local_work_size,
                                 0, NULL, &velocity_trigger[0]);

   if (ret != CL_SUCCESS) {
        printf("Error executing kernel Here1: %d\n", ret);
    }   


    ret = clEnqueueNDRangeKernel(info.command_queue, info.velocity_computek, 2,
                                 work_offset, global_work_size, local_work_size,
                                 1, &velocity_trigger[0], &velocity_complete);
    if (ret != CL_SUCCESS) {
        printf("Error executing kernel Here2: %d\n", ret);
    }


    ret = clEnqueueReadBuffer(info.command_queue, velocity_obj, CL_TRUE, 0, cl_float2_padded_screen_size, velocity, 0, NULL, NULL);

    if (ret != CL_SUCCESS) {
        fprintf(stderr, "Unable to read data. %d\n", ret);
    }


    //printf("Read results...\n");

    ret = clFlush(info.command_queue);
    clReleaseMemObject(velocity_obj);
    //clReleaseMemObject(velocity1_obj);
    //clReleaseMemObject(velocity2_obj);
    clReleaseMemObject(impossible_obj);
    clReleaseMemObject(pressure_obj);
    clReleaseMemObject(pressure2_obj);
    clReleaseMemObject(pressure_gradient_obj);
    clReleaseMemObject(divergence_obj);

    clReleaseEvent(writes_complete[0]);
    clReleaseEvent(writes_complete[1]);
    //clReleaseEvent(init_done[0]);
    //clReleaseEvent(init_done[1]);
    clReleaseEvent(divergence_complete);
    clReleaseEvent(velocity_trigger[0]);
//    clReleaseEvent(pressure_complete);
//    clReleaseEvent(velocity_complete);

    //printf("Didnt segfault\n");

}

void 
fluid_compute_concentration(float        *concentration,
                            cl_float2    *velocity,
                            char         *impossible_list,
                            int           width,
                            int           height,
                            float         d_coeff,
                            float         tile_distance,
                            float         time_interval,
                     struct fluid_cl_info info)
{

    cl_mem concentration_obj, velocity_obj, impossible_obj;

    cl_event diffuse_done;

    cl_int ret;

    concentration_obj = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(float)*(height)*(width), NULL, &ret);
    velocity_obj      = clCreateBuffer(info.context, CL_MEM_READ_ONLY, cl_float2_padded_screen_size, NULL, &ret);
    impossible_obj    = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(char)*(height+2)*(width+2), NULL, &ret);

    fluid_clSetKernelArg(info.diffuse_computek, 0, sizeof(cl_mem), (void *)&concentration_obj, ret);
    fluid_clSetKernelArg(info.diffuse_computek, 1, sizeof(cl_mem), (void *)&velocity_obj, ret);
    fluid_clSetKernelArg(info.diffuse_computek, 2, sizeof(cl_mem), (void *)&impossible_obj, ret);
    fluid_clSetKernelArg(info.diffuse_computek, 3, sizeof(int), (void *)&width, ret);
    fluid_clSetKernelArg(info.diffuse_computek, 4, sizeof(int), (void *)&height, ret);
    fluid_clSetKernelArg(info.diffuse_computek, 5, sizeof(float), (void *)&d_coeff, ret);
    fluid_clSetKernelArg(info.diffuse_computek, 6, sizeof(float), (void *)&time_interval, ret);
    fluid_clSetKernelArg(info.diffuse_computek, 7, sizeof(float), (void *)&tile_distance, ret);


    ret = clEnqueueWriteBuffer(info.command_queue, concentration_obj, CL_TRUE, 0, sizeof(float)*(height)*(width), concentration, 0, NULL, NULL); 
    ret = clEnqueueWriteBuffer(info.command_queue, velocity_obj, CL_TRUE, 0, cl_float2_padded_screen_size, velocity, 0, NULL, NULL);
    ret = clEnqueueWriteBuffer(info.command_queue, impossible_obj, CL_TRUE, 0, sizeof(char)*(height+2)*(width+2), impossible_list, 0, NULL, NULL);


    size_t work_offset[2] = {0, 0};
    size_t global_work_size[2] = {width, height};
    size_t local_work_size[2] = {8, 8};

    clEnqueueNDRangeKernel(info.command_queue, info.diffuse_computek, 2,
                           work_offset, global_work_size, local_work_size,
                           0, NULL, &diffuse_done);

    ret = clEnqueueReadBuffer(info.command_queue, concentration_obj, CL_TRUE, 0, sizeof(float)*(height)*(width), concentration, 1, &diffuse_done, NULL);

    if (ret != CL_SUCCESS) {
        printf("Failure to read: %d\n", ret);
    }

    ret = clFlush(info.command_queue);
    clReleaseMemObject(concentration_obj);
    clReleaseMemObject(velocity_obj);
    clReleaseMemObject(impossible_obj);

}

