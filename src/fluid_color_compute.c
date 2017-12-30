
#include "fluid_cl.h"

#include <CL/cl.h>

void
fluid_compute_pixels(int            *pixels,
                     float          *concentration,
                     int             width,
                     int             height,
                     float           scale,
              struct fluid_cl_info   info)
{
    cl_mem concentration_obj, hue_obj, pixels_obj;

    cl_int ret;

    cl_event hue_complete, pixels_complete;

    concentration_obj = clCreateBuffer(info.context, CL_MEM_READ_ONLY, sizeof(float)*width*height, NULL, &ret);
    hue_obj = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(float)*width*height, NULL, &ret);
    pixels_obj = clCreateBuffer(info.context, CL_MEM_READ_WRITE, sizeof(int)*width*height, NULL, &ret);

    fluid_clSetKernelArg(info.hue_computek, 0, sizeof(cl_mem), (void *)&concentration_obj, ret);
    fluid_clSetKernelArg(info.hue_computek, 1, sizeof(cl_mem), (void *)&hue_obj, ret);
    fluid_clSetKernelArg(info.hue_computek, 2, sizeof(float), (void *)&scale, ret);
    fluid_clSetKernelArg(info.hue_computek, 3, sizeof(int), (void *)&width, ret);
    fluid_clSetKernelArg(info.hue_computek, 4, sizeof(int), (void *)&height, ret);

    fluid_clSetKernelArg(info.pixels_computek, 0, sizeof(cl_mem), (void *)&hue_obj, ret);
    fluid_clSetKernelArg(info.pixels_computek, 1, sizeof(cl_mem), (void *)&pixels_obj, ret);
    fluid_clSetKernelArg(info.pixels_computek, 2, sizeof(int), (void *)&width, ret);
    fluid_clSetKernelArg(info.pixels_computek, 3, sizeof(int), (void *)&height, ret);

    ret = clEnqueueWriteBuffer(info.command_queue, concentration_obj, CL_TRUE, 0, sizeof(float)*width*height, concentration, 0, NULL, NULL);

    size_t work_offset[2] = {0, 0};
    size_t global_work_size[2] = {width, height};
    size_t local_work_size[2] = {8, 8};

    ret = clEnqueueNDRangeKernel(info.command_queue, info.hue_computek, 2,
                                 work_offset, global_work_size, local_work_size,
                                 0, NULL, &hue_complete);

    ret = clEnqueueNDRangeKernel(info.command_queue, info.pixels_computek, 2,
                                 work_offset, global_work_size, local_work_size,
                                 1, &hue_complete, &pixels_complete);

    ret = clEnqueueReadBuffer(info.command_queue, pixels_obj, CL_TRUE, 0, sizeof(int)*width*height, pixels, 1, &pixels_complete, NULL);

    if (ret != CL_SUCCESS) {
        printf("Could not read buffer: %d\n", ret);
    }

    ret = clFlush(info.command_queue);
    clReleaseMemObject(concentration_obj);
    clReleaseMemObject(hue_obj);
    clReleaseMemObject(pixels_obj);

    clReleaseEvent(hue_complete);
    clReleaseEvent(pixels_complete);


}

