#include "World.h"
#include "Field.h"
#include "Vec3.h"

int main(int argc, char *args[])
{
    World world(/*...*/); // initialize computational domain

    Species ions(/*...*/); // initialize particle species
    Species electrons(/*...*/);

    SolvePotential();       // get initial field
    ComputeElectricField(); // differentiate potential

    GenerateParticles(); // introudce particles

    // main loop
    for (ts = 0; ts < num_ts; ts++)
    {
        ComputeChargeDensity(); // scatter particle positions

        SolvePotential();       // solve Poisson's equation
        ComputeElectricField(); // differentiate potential

        IntegrateVelocity(); // advance particle velocity
        IntegratePosition(); // advance particle position

        RunTimeDiagnostics(); // placeholder for diagnostics
    }

    OutputResults(); // save results to the disc
    return 0;        // normal exit
}