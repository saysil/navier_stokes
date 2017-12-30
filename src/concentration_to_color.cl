
__kernel void 
concentration_to_hue(__global float *species,
                     __global float *hue,
                              float scale,
                              int   width,
                              int   height)
{

    int j = get_global_id(0);
    int i = get_global_id(1);

    /* 240 degrees - value/scale * 120 degrees */
    hue[i * width + j] = fmod((float)((4.0f/3.0f) * M_PI + species[i*width + j]/scale * ((2.0f/3.0f) * M_PI)), (float)(2.0f*M_PI));
}

__kernel void
hue_to_rgb(__global float *huearray,
           __global uint  *pixels_rgb,
                    int    width,
                    int    height)
{

    int j = get_global_id(0);
    int i = get_global_id(1);

    // X = (1 - |(H/(pi/3)) mod 2 - 1|)
    float hue = huearray[i*width + j]/(M_PI/3.0f);
    float X = 1.0f - fabs((float)(fmod((float)(hue), 2.0f) - 1.0f));
    char r, g, b, a;

    r = (1.0f*(0.0f <= hue)*(hue < 1.0f) + 1.0f*(5.0f <= hue)*(hue < 6.0f) + X*(1.0f <= hue)*(hue < 2.0f) + X*(4.0f <= hue)*(hue < 5.0f));

    g = (X*(0.0f <= hue)*(hue < 1.0f) + 1.0f*(1.0f <= hue)*(hue <= 3.0f) + X*(3.0f <= hue)*(hue < 4.0f));

    b = (X*(2.0f <= hue)*(hue < 3.0f) + 1.0f*(3.0f <= hue)*(hue < 5.0f) + X*(5.0f <= hue)*(hue <= 6.0f));


    a = 0;

    pixels_rgb[i * width + j] = (int)((r*255) << 24) + ((int)(g*255) << 16) + ((int)(b*255) << 8) + (int)a;

}

