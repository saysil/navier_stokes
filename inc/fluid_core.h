
/* Core dynamics of tile fluids, defines a vector containing all fluid tiles */ 

#ifndef FLUID_CORE_H
#define FLUID_CORE_H

struct fluid_core {
    float temperature; //kelvin
    float moles;       //moles of fluid (total)

    float volume;      //volume available for gasses

};

#endif

