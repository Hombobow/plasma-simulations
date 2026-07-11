#include "initialization.h"
#include "particle.h"
#include "grid.h"
#include <iostream>
#include <string>
#include <cmath>
#include <random>
#include <fstream>
#include <sstream>
#include <iomanip>


using namespace std;
int scaling = 4; 
int n_steps; 
int N_ppc; 
int n_cells; 
int n_nodes; 
int n_particles; 
double L; 
double dt; 
double dx ; 
double T; 
double w; // macroparticle weight
std::vector<Particle> electrons; 
GridData datagrid; 
std::string problem; 

void initialize(){
    problem = "landaudamping"; // or "twostream"
    L = 2 * pi; 
    n_cells = 32 * scaling; 
    n_nodes = n_cells + 1 ; 
    T = 30; 
    dt = 0.04 / scaling; 
    dx = L / n_cells ; 
    n_steps = static_cast<int>(std::round(T / dt));; 
    
    vector<double> n_e_init(n_cells); 
    vector<double> n_i_init(n_cells); 
    vector<double> phi_init(n_cells); 
    vector<double> x_cells_init(n_cells); 
    vector<double> x_nodes_init(n_nodes); 
    vector<double> E_init(n_nodes); // stored on nodes

    if (problem == "landaudamping"){
        N_ppc = 100000; 


        // initialize grid data
        double delta_n = 0.3;  
        for (int i = 0; i < n_cells; i++){
            x_cells_init[i] = i * dx + dx / 2; 
            x_nodes_init[i] = i * dx; 

            n_i_init[i] = 1; 
            n_e_init[i] = 1 + delta_n * sin(x_cells_init[i]) ; 
            phi_init[i] = -delta_n * sin(x_cells_init[i]) ; // this isn't used
            E_init[i] = - delta_n * cos(x_nodes_init[i]);// this isn't used

            // get phi_init, E_init from the initial density perturbation
        }
        x_nodes_init[n_nodes -1] = L; 
        E_init[n_nodes -1] = - delta_n * cos(L);

        datagrid = GridData(n_cells, n_e_init, n_i_init, phi_init, E_init, x_cells_init, x_nodes_init); 
        datagrid.poisson_solve();   // recomputes phi from the initial density, with phi[0] = 0
        datagrid.update_E();        // and the matching initial E

        // initialize electrons

        w = ni / N_ppc; 
        n_particles = N_ppc * n_cells; 
        electrons.resize(n_particles);
        
        random_device rd; std::mt19937 gen(rd()); std::uniform_real_distribution<double> dis(0.0, 1.0);

        for (int i = 0; i < n_particles; i++){
            while(true){
                double R1 = dis(gen); 
                double R2 = dis(gen); 
                if (R2 * 1.3 < 1.0 + 0.3 * sin(2*pi * R1 / 2 )){ // divide by 2 to have a smaller wavenumber
                    electrons[i].x = 2 * pi * R1 ; 
                    double R3 = dis(gen); 
                    double R4 = dis(gen); 
                    // replace with initializer
                    electrons[i].vx = sqrt(-2 * log(R3)) * cos(2 * pi * R4)  - electrons[i].get_q()/electrons[i].get_m() *  dt/2 * get_E(2 * pi * R1); 
                    // note 1: the PDF said 1, I used 2 because that's what the Box-Muller transform is online
                    // note 2: vx needs to be initialized @ t = -dt/2
                    break; 
                }
            }
            
        }


    }
    else{ // if problem == "twostream"

    }
}


int gather_density(){

    double part_x, f; int ind_minus, ind_plus, i; 
    for ( i = 0; i < n_cells; i++)    datagrid.n_e[i] = 0;
    for ( i = 0; i < n_particles; i++){
        part_x = electrons[i].get_x(); 
        
        if (part_x < dx/2) {
            ind_minus = n_cells - 1; 
            ind_plus = 0; 
            f = (datagrid.x_cells[ind_plus] - part_x ) / dx; // fraction deposited to minus cell
        }
        else if (part_x > L - dx/2){
            ind_minus = n_cells - 1; 
            ind_plus = 0; 
            f = (datagrid.x_cells[ind_plus] - part_x +L) / dx; // fraction deposited to minus cell

            
        }
        else{
            ind_minus = (part_x - dx/2) / dx; 
            ind_plus = ind_minus + 1; 
            f = (datagrid.x_cells[ind_plus] - part_x   ) / dx; // fraction deposited to minus cell
        }
        
        datagrid.n_e[ind_minus] += w * f; 
        datagrid.n_e[ind_plus] += w * (1-f); 
    }

    return 1; 
}


int push_electrons(){
    for (int i = 0; i < n_particles; i++){
        double F = electrons[i].get_q() / electrons[i].get_m() * get_E(electrons[i].get_x()); 
        electrons[i].update_vx(dt, F); 
        electrons[i].update_x(dt, L); 
    }
    return 1; 
}
int output_data(int n_output){

    {
        std::ostringstream name;
        name << "output/particles/particle_" << n_output << ".csv";
        std::ofstream f(name.str());
        f << std::setprecision(9);
        f << "x,vx\n";
        for (int i = 0; i < n_particles; i++){
            f << electrons[i].get_x() << "," <<electrons[i].get_vx() + electrons[i].get_q() / electrons[i].get_m() * (dt / 2) * get_E(electrons[i].get_x()) << "\n";
        }
    }

    {
        std::ostringstream name;
        name << "output/fields/fields_" << n_output << ".csv";
        std::ofstream f(name.str());
        f << std::setprecision(9);
        f << "x,n,E,phi\n";
        for (int i = 0; i < n_cells; i++){
            double E_cell = 0.5 * (datagrid.E[i] + datagrid.E[i + 1]);
            f << datagrid.x_cells[i] << "," << datagrid.n_e[i] << ","
              << E_cell << "," << datagrid.phi[i] << "\n";
        }
    }

    {
        double ES = datagrid.get_E_energy();       
        double KE = 0.0;
        for (int i = 0; i < n_particles; i++){
            double v = electrons[i].get_vx() + electrons[i].get_q() / electrons[i].get_m() * (dt / 2) * get_E(electrons[i].get_x()) ;
            KE += 0.5 * w * dx * electrons[i].get_m() * v * v;
        }
        std::ostringstream name;
        name << "output/scalars/scalars_" << n_output << ".csv";
        std::ofstream f(name.str());
        f << std::setprecision(9);
        f << "ES_energy,KE\n";
        f << ES << "," << KE << "\n";
    }

    return 1;
}
int run_loop(){
    int nt; 
    
    int tenth = n_steps / 10; if (tenth < 1) tenth = 1;   // progress bar: ~one block per 10%

    for (nt = 0; nt < n_steps; nt++){
        if (nt % tenth == 0) std::cout << "[]" << std::flush;
        if (nt % scaling == 0){
            output_data(nt/scaling);

        }

        push_electrons(); 
        gather_density(); 
        datagrid.poisson_solve(); 
        datagrid.update_E(); 
        

    }

    return 1; 
}

double get_E(double part_x){
    int ind_minus = part_x / dx; 
    int ind_plus = ind_minus + 1; 

    return datagrid.E[ind_minus] * (datagrid.x_nodes[ind_plus] - part_x )/dx + datagrid.E[ind_plus]* (part_x - datagrid.x_nodes[ind_minus]  )/dx ; 
}
