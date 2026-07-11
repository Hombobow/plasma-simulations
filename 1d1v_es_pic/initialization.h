#ifndef INITIALIZATION_H
#define INITIALIZATION_H
// this file initializes global constants and functions
#include "particle.h"
#include "grid.h"
#include <string> 
#include <vector> 

const double pi = 3.141592653589793238; 
const double ni = 1.0;

extern int n_steps; 
extern int N_ppc; 
extern int n_cells; 
extern int n_nodes; 
extern int n_particles; 
extern double L; 
extern double dt; 
extern double dx ; 
extern double T; 
extern double w; // macroparticle weight
extern std::vector<Particle> electrons; 
extern GridData datagrid; 
extern std::string problem; 



void initialize(); 
int gather_density(); 
int push_electrons();
int output_data(int n_outpu); 
int run_loop(); 
double get_E(double x_particle) ; 

#endif // for INITIALIZATION_H