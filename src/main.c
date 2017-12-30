
#include <stdio.h>
#include <string.h>

#include <GL/gl.h>
#include <GL/glu.h>

#include <GL/glut.h>
#include "fluid_cl.h"

struct fluid_cl_info cl_info;

float *concentration;
cl_float2 *velocity;
char  *impossible_list;
struct fluid_core *fluids;
int   *pixels;

#define _ij(k, z) ((k+1)*(1920+2) + (z+1))

void tmpdiffuse(cl_float2 *velocity1, cl_float2 *velocity0) 
{
    float a = 20.0f*0.1f/(0.05f*0.05f);
    for (int k=0; k<20; k++) {
    for (int i=0; i<1080; i++) {
        for (int j=0; j<1920; j++) {
            velocity1[_ij(i, j)].s[0] = (velocity0[_ij(i, j)].s[0] + a*(velocity1[_ij(i-1, j)].s[0] + velocity1[_ij(i+1, j)].s[0]
                                                                   + velocity1[_ij(i, j-1)].s[0] + velocity1[_ij(i, j+1)].s[0]))/(1.0f+4.0f*a);
        
            velocity1[_ij(i, j)].s[1] = (velocity0[_ij(i, j)].s[1] + a*(velocity1[_ij(i-1, j)].s[1] + velocity1[_ij(i+1, j)].s[1]
                                                                   + velocity1[_ij(i, j-1)].s[1] + velocity1[_ij(i, j+1)].s[1]))/(1.0f+4.0f*a); 

        }
    }
    }
}

void add_moles(float *_concentration, struct fluid_core *_fluids)
{
    for (int i=0; i<1080; i++) {
        for (int j=0; j<1920; j++) {
            _fluids[i*1920 +j].moles = _concentration[i*1920 + j];
        }
    }
}

void on_mouse(int button, int state, int x, int y) 
{

    //printf("Pressed.\n");
    if (button == GLUT_LEFT_BUTTON) {
        //fluids[(1080-y)*1920 + x].moles += 0.05f;
        //concentration[(1080-y)*1920 + x] += 0.05f;
        printf("V-Vector@(%d, %d): [%f, %f]\n", (1080-y), x, velocity[(1080-y)*1922 + x + 1].s[0], velocity[(1080-y)*1922 + x + 1].s[1]);
    }
}

void display(void)
{
    //printf("Concentration: %f\nVelocity: %f, %f", concentration[400*1920 + 2 - 1], velocity[401*1922 + 2 - 1].s[0], velocity[401*1922 + 2 - 1].s[1]);

    ///printf("Computing velocity...\n");
    fluid_compute_velocity(velocity, impossible_list, 1920, 1080, 0.05f, 0.8f, 1.25f, cl_info);
    //cl_float2 *v0 = malloc(1922*1082*sizeof(cl_float2));
    //memcpy(v0, velocity, 1922*1082*sizeof(cl_float2));
    //tmpdiffuse(velocity, v0);
    //free(v0);
    //printf("Diffusing...\n"); 
    fluid_compute_concentration(concentration, velocity, impossible_list, 1920, 1080, 1.0f, 0.05, 0.016, cl_info);
    //printf("Drawing...\n");
    fluid_compute_pixels(pixels, concentration, 1920, 1080, 1.0f, cl_info);
    glDrawPixels(1920, 1080, GL_RGBA, GL_UNSIGNED_INT_8_8_8_8, pixels);
    //add_moles(concentration, fluids);
    glutSwapBuffers();
 
}

void
stokes_timer_function(int value)
{
    glutPostRedisplay();
    glutTimerFunc(16, stokes_timer_function, 0);
}

void
ogl_init(void)
{
    glClearColor(0.0, 0.0, 0.0, 0.0);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, 1.0, 0.0, 1.0, -1.0, 1.0);
}

int 
main(int    argc,
     char **argv)
{

    cl_info = fluid_cl_init();

    printf("Allocating fluid values...\n");

    concentration = calloc(1920*1080, sizeof(float));
    velocity      = calloc((1920+2)*(1080+2), sizeof(cl_float2));
    impossible_list = calloc((1920+2)*(1080+2), sizeof(char));
    fluids          = calloc(1920*1080, sizeof(struct fluid_core));
    pixels          = calloc(1920*1080, sizeof(int));

    printf("Populating fluid values...\n");

    for (int i=0; i<1080; i++) {
        for (int j=0; j<1920; j++) {
            fluids[i*1920 + j].temperature = 40.0f;
            fluids[i*1920 + j].volume = 0.000125f;
            fluids[i*1920 + j].moles = 0.0f;
        }
    }

    fluids[200*1920 + 400].moles = 0.5f;
    concentration[200*1920 + 304] = 0.5f;
    fluids[201*1920 + 400].moles = 0.5f;
    concentration[201*1920 + 304] = 0.5f;

    for (int i=0; i<10; i++) {
        for (int j=0; j<10; j++) {
            velocity[(i+1+500)*(1920+2) + j+200 + 1].s[0] = 0.0f;
            velocity[(i+1+500)*(1920+2) + j+200 + 1].s[1] = 0.5f;
        }
    }

    printf("Starting OpenGL Routines...\n");

    glutInit(&argc, argv);;
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(1920, 1080);
    glutInitWindowPosition(-1, -1);
    glutCreateWindow("Navier Stokes Simulation: 1920x1080");
    ogl_init();
    glutMouseFunc(on_mouse);
    glutDisplayFunc(display);
    stokes_timer_function(0); 
    glutMainLoop();

    return 0;
}

