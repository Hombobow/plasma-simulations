#ifndef GRID_H
#define GRID_H
#include <eigen3/Eigen/Dense>
#include <vector>
class GridData{
    private:
        double E_energy;
        std::vector<std::vector<double>> A_inv;
        Eigen::MatrixXd A; 
        std::vector<double> rho; 

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