#ifndef INITIALIZATION_H
#define INITIALIZATION_H

// this file initializes global constants and functions
#include "grid.h"
#include "particle.h"
#include <string>
#include <vector>

const double pi = 3.141592653589793238; 
const double ni = 1.0;

extern int n_steps; // number of time steps
extern int N_ppc; // particles per cell
extern int n_cells; // number of cells
extern int n_nodes; // number of nodes
extern int n_particles; // number of particles (N_ppc * n_cells)
extern double L; // domain size
extern double dt; // time step
extern double dx; // cell size 
extern double T; // total time
extern double w; // macroparticle weight
extern std::vector<Particle> electrons; 
extern GridData datagrid; 
extern std::string problem; 
extern std::string output_dir;  // e.g. "output/twostream_wl2"

void initialize();
int gather_density();
int push_electrons();
int output_data(int n_output);
int run_loop();
double get_E(double x_particle);

#endif