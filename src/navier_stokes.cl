
/* This function modifies the velocity vector field of *
 * ALL tiles on the map in accordance with the Navier- *
 * Stokes equations of fluid dynamics. This function   *
 * should be called before the the diffusion function  *
 * 240x135 work groups 8x8 work items for 1920x1080    */

#include "fluid_core.h"

#define one_abs(x) ((x)*(x))


/* For Clarity, let [_ij] be the common coordinates [i][j]. 2D arrays are not supported in Opencl. */
#define _ij(k, z)  ((k)*(width+2) + (z))
#define nopad_ij(k, z) (((k)-1)*(width) + ((z)-1))


//#define can_contain_fluid(k, z) (((k) != 0)*((k) != height)*((z) != 0)*((z) != width)*(1.0f-(float)fluid_impossible_list[_ij(k, z)]))

__kernel void
fluid_diffuse_velocity(__global float2 *velocity_initial, //padded
                       __global float2 *velocity_copy, //padded,
                       __global float2 *velocity_new, //padded,
                       __global char   *fluid_impossible_list, //padded
                                int     width,
                                int     height,
                                float   viscosity,
                                float   tile_distance,
                                float   time_interval)
{
    int j = get_global_id(0) + 1;
    int i = get_global_id(1) + 1;

    float a = time_interval*viscosity/(tile_distance*tile_distance);

    // Jacobi iteration in Joe Stam's code for fast fluid simulations.

    velocity_new[_ij(i, j)].x = (velocity_initial[_ij(i, j)].x
                                + a*(velocity_copy[_ij(i-1, j)].x + velocity_copy[_ij(i+1, j)].x +
                                     velocity_copy[_ij(i, j-1)].x + velocity_copy[_ij(i, j+1)].x))/(1.0f + 4.0f*a);


    velocity_new[_ij(i, j)].y = (velocity_initial[_ij(i, j)].y
                                + a*(velocity_copy[_ij(i-1, j)].y + velocity_copy[_ij(i+1, j)].y +
                                     velocity_copy[_ij(i, j-1)].y + velocity_copy[_ij(i, j+1)].y))/(1.0f + 4.0f*a);


}

__kernel void
fluid_convect_velocity(__global float2 *velocity, //padded
                       __global float2 *velocity_new,
                                int     width,
                                int     height,
                                float   tile_distance,
                                float   time_interval)
{
    int j = get_global_id(0) + 1;
    int i = get_global_id(1) + 1;

    float a = velocity[_ij(i, j)].x;
    float b = velocity[_ij(i, j)].y;
    velocity_new[_ij(i, j)].x = a - (a*(velocity[_ij(i+1, j)].x - velocity[_ij(i-1, j)].x)/(2*tile_distance)
                                  +  b*(velocity[_ij(i, j+1)].x - velocity[_ij(i, j-1)].x)/(2*tile_distance))*time_interval;

    velocity_new[_ij(i, j)].y = b - (a*(velocity[_ij(i+1, j)].y - velocity[_ij(i-1, j)].y)/(2*tile_distance)
                                  +  b*(velocity[_ij(i, j+1)].y - velocity[_ij(i, j-1)].y)/(2*tile_distance))*time_interval;

}
                       

__kernel void
fluid_calculate_divergence(__global float2 *velocity, //padded
                           __global float  *divergence, //not padded
                                    int     width,
                                    int     height,
                                    float   tile_distance)
{
    int j = get_global_id(0) + 1;
    int i = get_global_id(1) + 1;
    divergence[nopad_ij(i, j)] = 0.5f * (velocity[_ij(i, j+1)].x  - velocity[_ij(i, j-1)].x + 
                                         velocity[_ij(i-1, j)].y - velocity[_ij(i+1, j)].y)/tile_distance;
    
    //divergence[nopad_ij(i, j)] = velocity[_ij(i-1, j)].y;
}

__kernel void
fluid_pressure_field(__global float2            *velocity,       //not padded
                     __global float             *pressure_field, //will be padded
                     __global float             *pressure_field_new,
                     __global float             *divergence,
                              int                width,
                              int                height,
                              float              tile_distance,
                              float              time_interval)
{
    int j = get_global_id(0) + 1; //horizontal thread + 1
    int i = get_global_id(1) + 1; //vertical thread + 1

    /* Run relaxation iterations for poissons equation */
    pressure_field_new[_ij(i, j)] = 0.25f * (pressure_field[_ij(i, j+1)] + pressure_field[_ij(i, j-1)]
                                  + pressure_field[_ij(i+1, j)] + pressure_field[_ij(i-1, j)]
                                  - tile_distance*tile_distance*1.2/time_interval*divergence[nopad_ij(i, j)]);
   

//   pressure_field_new[_ij(i, j)] = divergence[nopad_ij(i, j)];
}

__kernel void 
fluid_pressure_gradient(__global float   *pressure_field, //must be padded
                        __global float2  *pressure_gradient, //not padded
                                 int      width,
                                 int      height,
                                 float    tile_distance)
{


    int j = get_global_id(0) + 1;
    int i = get_global_id(1) + 1;



    pressure_gradient[nopad_ij(i, j)].x = 0.5f * (pressure_field[_ij(i, j+1)] - pressure_field[_ij(i, j-1)])/tile_distance;
    pressure_gradient[nopad_ij(i, j)].y = 0.5f * (pressure_field[_ij(i-1, j)] - pressure_field[_ij(i+1, j)])/tile_distance;

}

__kernel void 
fluid_velocity_calculate(__global float2 *velocity, //must be padded
                         __global char   *fluid_impossible_list, //also must be padded
                         __global float2 *pressure_gradient,
                                  float   viscosity,
                                  int     width,
                                  int     height,
                                  float   time_interval)
{
    int j = get_global_id(0) + 1;
    int i = get_global_id(1) + 1;

    /* Make sure no velocity vector is pointing inside of a wall */


    velocity[_ij(i, j)].x += (-time_interval/1.2*pressure_gradient[nopad_ij(i, j)].x)*time_interval;
    velocity[_ij(i, j)].y += (-time_interval/1.2*pressure_gradient[nopad_ij(i, j)].y)*time_interval;

    /* If we can't contain fluid, theres obviously no velocity */
//    velocity[_ij(i, j)] *= (1.0f-fluid_impossible_list[_ij(i, j)]);
//    velocity[_ij(i, j)] = pressure_gradient[_ij(i-1, j)];
}

