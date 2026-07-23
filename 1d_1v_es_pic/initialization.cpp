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
#include <cstdlib>
#include <omp.h>

using namespace std;
int scaling = 1;
int n_steps;
int N_ppc;
int n_cells;
int n_nodes;
int n_particles;
double L;
double dt;
double dx;
double T;
double w; // macroparticle weight

vector<Particle> electrons;
GridData datagrid;
string problem;
string output_dir;

void initialize() {
    double wavelength = 2.0; // in units of 2*pi; check: 1, 2, 3
    double k = 1.0 / wavelength;
    problem = "landaudamping";                              // "landaudamping" or "twostream"
    output_dir = "output/" + problem;                   // "output/landaudamping or output/twostream"
    L = wavelength * 2 * pi;                            // must be an integer multiple of 2*pi/k
    n_cells = 32 * scaling * (int)round(L / (2 * pi));  // scale cells with box so dx (resolution) stays fixed as you change L
    n_nodes = n_cells + 1;
    T = 30;
    dt = 0.04 / scaling;
    dx = L / n_cells;
    n_steps = static_cast<int>(round(T / dt));

    vector<double> n_e_init(n_cells);
    vector<double> n_i_init(n_cells);
    vector<double> phi_init(n_cells);
    vector<double> x_cells_init(n_cells);
    vector<double> x_nodes_init(n_nodes);
    vector<double> E_init(n_nodes); // stored on nodes

    if (problem == "landaudamping")
    {
        N_ppc = 10000;

        // initialize grid data
        double delta_n = 0.05;
        double ne0 = 1; //+ delta_n / (L*k) *(cos(k*L) - 1) ;
        for (int i = 0; i < n_cells; i++)
        {
            x_cells_init[i] = i * dx + dx / 2;
            x_nodes_init[i] = i * dx;

            n_i_init[i] = 1; // (or ne0, a fixed number) — a static background array
            n_e_init[i] = ne0 + delta_n * sin(x_cells_init[i] * k);
            phi_init[i] = -delta_n / (k * k) * sin(k * x_cells_init[i]);
            E_init[i] = delta_n / k * cos(k * x_nodes_init[i]);
        }
        x_nodes_init[n_nodes - 1] = L;
        E_init[n_nodes - 1] = delta_n / k * cos(k * x_nodes_init[n_nodes - 1]); // last node (x = L), same analytic E

        datagrid = GridData(n_cells, n_e_init, n_i_init, phi_init, E_init, x_cells_init, x_nodes_init);
        datagrid.poisson_solve(); // JIC
        datagrid.update_E();      // JIC

        // 1D macroparticle weight: physical particles per macro = n0*dx / N_ppc
        w = ni * dx / N_ppc;
        // density pert. can put more than N_ppc in a cell — grow with push_back, then set n_particles
        electrons.clear();
        electrons.reserve(static_cast<size_t>(N_ppc * n_cells * (1.0 + delta_n) + n_cells));

        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<double> dis(0.0, 1.0);

        for (int i = 0; i < n_cells; i++)
        {
            int num_electrons = static_cast<int>(std::round(
                N_ppc * (1.0 + delta_n * sin(k * datagrid.x_cells[i]))));

            for (int j = 0; j < num_electrons; j++) {
                double R1 = dis(gen);
                double R2 = std::max(dis(gen), 1e-300); // avoid log(0)
                double R3 = dis(gen);
                Particle p;

                p.x = datagrid.x_nodes[i] + R1 * dx;
                p.vx = sqrt(-2.0 * log(R2)) * cos(2.0 * pi * R3);
                electrons.push_back(p);
            }
        }
        n_particles = static_cast<int>(electrons.size());
        cout << "=== landaudamping run ===\n"
                  << "  problem      = " << problem << "\n"
                  << "  scaling      = " << scaling << "\n"
                  << "  wavelength   = " << wavelength << " (units of 2*pi)\n"
                  << "  k            = " << k << "\n"
                  << "  L            = " << L << "\n"
                  << "  n_cells      = " << n_cells << "\n"
                  << "  n_nodes      = " << n_nodes << "\n"
                  << "  dx           = " << dx << "\n"
                  << "  dt           = " << dt << "\n"
                  << "  T            = " << T << "\n"
                  << "  n_steps      = " << n_steps << "\n"
                  << "  N_ppc        = " << N_ppc << "\n"
                  << "  n_particles  = " << n_particles
                  << "  (N_ppc*n_cells=" << N_ppc * n_cells << ")\n"
                  << "  w            = " << w << "  (ni*dx/N_ppc)\n"
                  << "  ne0          = " << ne0 << "\n"
                  << "  delta_n      = " << delta_n << "\n"
                  << "  ni           = " << ni << endl;
    }
    else if (problem == "twostream")
    { // if problem == "twostream"
        double factor = 4;
        N_ppc = 1000 * pow(2, factor);
        double v_th = 0.00;   // thermal velocity (normalized)
        double v_drift = 0.8; // drift velocity (must be > v_th for instability)

        // initialize grid data
        double delta_n = 0.001;
        double ne0 = 1; //+ delta_n / (L*k) *(cos(k*L) - 1) ;
        for (int i = 0; i < n_cells; i++)
        {
            x_cells_init[i] = i * dx + dx / 2;
            x_nodes_init[i] = i * dx;

            n_i_init[i] = ne0;
            n_e_init[i] = ne0 + delta_n * sin(x_cells_init[i] * k);
            phi_init[i] = -delta_n / (k * k) * sin(k * x_cells_init[i]);
            E_init[i] = delta_n / k * cos(k * x_nodes_init[i]);
        }
        x_nodes_init[n_nodes - 1] = L;
        E_init[n_nodes - 1] = delta_n / k * cos(k * x_nodes_init[n_nodes - 1]); // last node (x = L), same analytic E

        datagrid = GridData(n_cells, n_e_init, n_i_init, phi_init, E_init, x_cells_init, x_nodes_init);
        datagrid.poisson_solve(); // JIC
        datagrid.update_E();      // JIC

        // Must include dx so (w/dx)*f deposits number density ~ n0, and KE uses mass w.
        w = ni * dx / N_ppc;
        // density pert. can put more than N_ppc in a cell — grow with push_back, then set n_particles
        electrons.clear();
        electrons.reserve(static_cast<size_t>(N_ppc * n_cells * (1.0 + delta_n) + 2 * n_cells));

        random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<double> dis(0.0, 1.0);

        for (int i = 0; i < n_cells; i++)
        {
            int num_electrons = static_cast<int>(std::round(
                N_ppc * (1.0 + delta_n * sin(k * datagrid.x_cells[i])) / 2.0));

            for (int j = 0; j < num_electrons; j++)
            {
                double R1 = dis(gen);
                double R2 = std::max(dis(gen), 1e-300); // avoid log(0)
                double R3 = dis(gen);
                Particle p;
                p.x = datagrid.x_nodes[i] + dx * R1;
                p.vx = -v_drift + v_th * sqrt(-2.0 * log(R2)) * cos(2.0 * pi * R3) - p.get_q() / p.get_m() * dt / 2.0 * get_E(p.x);
                electrons.push_back(p);
            }

            for (int j = 0; j < num_electrons; j++)
            {
                double R1 = dis(gen);
                double R2 = std::max(dis(gen), 1e-300);
                double R3 = dis(gen);
                Particle p;
                p.x = datagrid.x_nodes[i] + datagrid.dx * R1;
                p.vx = v_drift + v_th * sqrt(-2.0 * log(R2)) * cos(2.0 * pi * R3) - p.get_q() / p.get_m() * dt / 2.0 * get_E(p.x);
                electrons.push_back(p);
            }
        }
        n_particles = static_cast<int>(electrons.size());
        cout << "=== twostream run ===\n"
                  << "  problem      = " << problem << "\n"
                  << "  scaling      = " << scaling << "\n"
                  << "  wavelength   = " << wavelength << " (units of 2*pi)\n"
                  << "  k            = " << k << "\n"
                  << "  L            = " << L << "\n"
                  << "  n_cells      = " << n_cells << "\n"
                  << "  n_nodes      = " << n_nodes << "\n"
                  << "  dx           = " << dx << "\n"
                  << "  dt           = " << dt << "\n"
                  << "  T            = " << T << "\n"
                  << "  n_steps      = " << n_steps << "\n"
                  << "  N_ppc        = " << N_ppc << "\n"
                  << "  n_particles  = " << n_particles
                  << "  (N_ppc*n_cells=" << N_ppc * n_cells << ")\n"
                  << "  w            = " << w << "  (ni*dx/N_ppc)\n"
                  << "  ne0          = " << ne0 << "\n"
                  << "  delta_n      = " << delta_n << "\n"
                  << "  ni           = " << ni << "\n"
                  << "  v_th         = " << v_th << "\n"
                  << "  v_drift      = " << v_drift << endl;
    }
}

int output_data(int n_output) {    
    if (n_output == 0) {
        {
            std::ostringstream name;
            name << output_dir << "/particles/particle_" << n_output << ".csv";
            std::ofstream f(name.str());
            f << std::setprecision(9);
            f << "x,vx\n";
            for (int i = 0; i < n_particles; i++)
            {
                f << electrons[i].get_x() << "," << electrons[i].get_vx() + electrons[i].get_q() / electrons[i].get_m() * (dt / 2) * get_E(electrons[i].get_x()) << "\n";
            }
        }
    } 

    // {
    //     std::ostringstream name;
    //     name << output_dir << "/fields/fields_" << n_output << ".csv";
    //     std::ofstream f(name.str());
    //     f << std::setprecision(9);
    //     f << "x,n,E,phi\n";
    //     for (int i = 0; i < n_cells; i++)
    //     {
    //         double E_cell = 0.5 * (datagrid.E[i] + datagrid.E[i + 1]);
    //         f << datagrid.x_cells[i] << "," << datagrid.n_e[i] << ","
    //             << E_cell << "," << datagrid.phi[i] << "\n";
    //     }
    // }

    {
        double ES = datagrid.get_E_energy();
        double KE = 0.0;
#pragma omp parallel for reduction(+ : KE)
        for (int i = 0; i < n_particles; i++)
        {
            double v = electrons[i].get_vx() + electrons[i].get_q() / electrons[i].get_m() * (dt / 2) * get_E(electrons[i].get_x());
            // macroparticle mass = w (= n0*dx/N_ppc in 1D); no extra dx
            KE += 0.5 * w * electrons[i].get_m() * v * v;
        }
        std::ostringstream name;
        name << output_dir << "/scalars/scalars_" << n_output << ".csv";
        std::ofstream f(name.str());
        f << std::setprecision(9);
        f << "ES_energy,KE\n";
        f << ES << "," << KE << "\n";
    }

    return 1;
}

static void clear_output_dirs()
{
    system(("rm -rf " + output_dir + "/particles " + output_dir + "/fields " + output_dir + "/scalars").c_str());
    system(("mkdir -p " + output_dir + "/particles " + output_dir + "/fields " + output_dir + "/scalars").c_str());
}

int run_loop()
{
    int nt;

    clear_output_dirs();

    int last_percent = -1;

    for (nt = 0; nt < n_steps; nt++)
    {
        int percent = (nt + 1) * 100 / n_steps;
        if (percent != last_percent)
        {
            cout << "\rProgress: " << setw(3) << percent << "%" << flush;
            last_percent = percent;
        }
        if (nt % scaling == 0)
        {
            output_data(nt / scaling);
        }
        push_electrons();
        gather_density();
        datagrid.poisson_solve();
        datagrid.update_E();
    }
    cout << "\n";

    return 1;
}

int gather_density() {
    for (int i = 0; i < n_cells; i++) {
        datagrid.n_e[i] = 0.0;
    }

#pragma omp parallel
    {
        vector<double> local_n_e(n_cells, 0.0); // private per thread

#pragma omp for nowait
        for (int i = 0; i < n_particles; i++) {
            double part_x = electrons[i].get_x();
            int ind_minus, ind_plus;
            double dist; // distance to the right (plus) cell 
            double f; // weighting fraction deposited to the left (minus) cell in linear (CIC) interpolation

            // new logic with modulo
            // in the case that x < dx / 2: x_shift < 0 and we get x_shift gets moved to the right end
            // in the case that x > L - dx / 2: dist gets an extra L
            double x_shift = part_x - dx / 2.0;
            if (x_shift < 0.0) x_shift += L;

            ind_minus = static_cast<int>(x_shift / dx); // 0 .. n_cells-1
            ind_plus = (ind_minus + 1) % n_cells;

            dist = datagrid.x_cells[ind_plus] - part_x;
            if (dist < 0.0) dist += L; // periodic wrap

            f = dist/dx;

            // w = n0*dx/N_ppc is a column density (particles / macroparticle);
            // divide by dx so the deposited cell value is a number density.
            local_n_e[ind_minus] += (w / dx) * f;
            local_n_e[ind_plus] += (w / dx) * (1 - f);
        }
#pragma omp critical
        for (int i = 0; i < n_cells; i++)
        {
            datagrid.n_e[i] += local_n_e[i];
        }
    }

    return 1;       
}

int push_electrons() {
#pragma omp parallel for
    for (int i = 0; i < n_particles; i++) {
        double F = electrons[i].get_q() / electrons[i].get_m() * get_E(electrons[i].get_x());
        electrons[i].update_vx(dt, F);
        electrons[i].update_x(dt, L);
    }
    return 1;
}

double get_E(double part_x) {
    int ind_minus = static_cast<int>(part_x / dx);
    int ind_plus = ind_minus + 1;
    return datagrid.E[ind_minus] * (datagrid.x_nodes[ind_plus] - part_x) / dx 
        + datagrid.E[ind_plus] * (part_x - datagrid.x_nodes[ind_minus]) / dx;
}