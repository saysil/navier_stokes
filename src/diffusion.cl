

/* Diffuse a chunk after defining a velocity vector */

#define height (__height)
#define width  (__width)
 
/* For Clarity, let [_ij] be the common coordinates [i][j]. 2D arrays are not supported in Opencl. */
#define _ij (i*(width+2) + j)

/* In all instances, i and j are the global ids PLUS 1 */
#define nopad_ij ((i-1)*(width) + j - 1)

#define tile_distance __tile_distance
#define time_interval __time_interval

//#define can_contain_fluid(k, n) (fluid_impossible_list[(k)*(width+2) + n])
#define can_contain_fluid(k, z) (((k) != 0)*((k) != height)*((z) != 0)*((z) != width)*(1.0f-(float)fluid_impossible_list[(k)*(width+2) + (z)]))


#define diffusion_coefficient __diffusion_coefficient

__kernel void 
fluid_diffuse_compute(__global float  *c, //not padded
                      __global float2 *V, //padded
                      __global char   *fluid_impossible_list, //must be padded
                               int     __width,
                               int     __height,
                               float   __diffusion_coefficient, 
                               float   __time_interval,
                               float   __tile_distance)
{

    int j = get_global_id(0) + 1;
    int i = get_global_id(1) + 1;

    /* Du/Dt = del * (D(x, y, t) * del u) */

    /* First things first, calculate the second derivatives for each tile that CAN have the gas */

    /* Recycled from the convective and laplacian form subroutine */
/*    float tmpx1, tmpx2, tmpy1, tmpy2 = 0;
 

    char x1_possible, x2_possible, y1_possible, y2_possible;
    x1_possible = can_contain_fluid(i, j-1);
    x2_possible = can_contain_fluid(i, j+1);

    y1_possible = can_contain_fluid(i-1, j);
    y2_possible = can_contain_fluid(i+1, j);

    tmpx1 = c[(i-1)*width + (j-1) - 1]*x1_possible;
    tmpx2 = c[(i-1)*width + (j-1) + 1]*x2_possible;

    tmpy1 = c[((i-1)-1)*width + (j-1)]*y1_possible;
    tmpy2 = c[((i-1)+1)*width + (j-1)]*y2_possible;



    #define NPOSSIBLE (x2_possible-x1_possible)
    #define POSSIBLE_ABS (1-(x2_possible-x1_possible)*(x2_possible-x1_possible))


    #define YNPOSSIBLE (y2_possible-y1_possible)
    #define YPOSSIBLE_ABS (1-(y2_possible-y1_possible)*(y2_possible-y1_possible))


*/

    /* dc/dt               =                          D * (dc/dx + dc/dy) */
    //c[nopad_ij] += (//diffusion_coefficient*(tmpx2 - 2*c[nopad_ij] + tmpx1)/((tile_distance)*(tile_distance)) +
                   //diffusion_coefficient*(tmpy2 - 2*c[nopad_ij] + tmpy1)/((tile_distance)*(tile_distance)) -
    /*                                                 del dot (v*c)      */
    //               -V[_ij].x*(tmpx2 - NPOSSIBLE*c[nopad_ij] - tmpx1)/((tile_distance) + (tile_distance)*POSSIBLE_ABS) - 
    //               V[_ij].y*(tmpy2 - YNPOSSIBLE*c[nopad_ij] - tmpy1)/((tile_distance) + (tile_distance)*YPOSSIBLE_ABS))*(time_interval);

   //c[nopad_ij] = (c[nopad_ij] == 0.0f);//NPOSSIBLE;

    c[nopad_ij] = 1000*sqrt(V[_ij].x*V[_ij].x + V[_ij].y*V[_ij].y);

//TODO: Currently, gasses can diffuse inside of walls and be destroyed, fix this
}


