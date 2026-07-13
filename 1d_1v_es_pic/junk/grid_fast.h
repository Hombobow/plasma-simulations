#ifndef GRID_H
#define GRID_H
#include <vector>
class GridData{
    private:
        double E_energy;
        std::vector<double> tridiagonal_solve(const std::vector<double>& a, const std::vector<double>& b,
                                         const std::vector<double>& c, const std::vector<double>& d);


        void tridiagonal_factor(const std::vector<double>& a, const std::vector<double>& b,
                                const std::vector<double>& c,
                                std::vector<double>& cp, std::vector<double>& inv_m);


        void tridiagonal_solve_factored(const std::vector<double>& a,
                                const std::vector<double>& cp, const std::vector<double>& inv_m,
                                const std::vector<double>& d,
                                std::vector<double>& dp, std::vector<double>& x);

        std::vector<double> pois_a, pois_b, pois_c;
        double pois_alpha = 1.0, pois_beta = 1.0;
        double pois_gamma = 0.0;
        std::vector<double> pois_bb;
        std::vector<double> pois_z;
        std::vector<double> pois_cp;      // precomputed factorization of A' (super-diagonal)
        std::vector<double> pois_inv_m;   // precomputed factorization of A' (reciprocal pivots)


        std::vector<double> pois_d;       // right-hand side
        std::vector<double> pois_dp;      // forward-sweep scratch
        std::vector<double> pois_y;       // first-solve result (A' y = d)
        bool pois_ready = false;

        void setup_poisson();
        void periodic_poisson_solve(const std::vector<double>& r, std::vector<double>& out);


    public: 
        GridData() = default; 
        GridData(const int n_cells_init, const std::vector<double> n_e_init, const std::vector<double> n_i_init, 
            const std::vector<double>phi_init, const std::vector<double> E_init, const std::vector<double> x_cells_init, const std::vector<double> x_nodes_init);
        std::vector<double> n_e;
        std::vector<double>n_i;  
        std::vector<double> phi; 
        std::vector<double> E; 
        std::vector<double> x_cells; 
        std::vector<double> x_nodes; 
        int n_cells;
        int n_nodes; 
        double dx; 

        int poisson_solve();
        int update_E();
        double get_E_energy();



};

#endif