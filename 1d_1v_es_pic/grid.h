// include guard
#ifndef GRID_H
#define GRID_H

// include directives
#include <eigen3/Eigen/Dense>
#include <vector>

// declarations go here (structs, classes, functions prototypes)
class GridData {
public:
    // constructor
    GridData() = default;
    GridData(const int& n_cells_init, const std::vector<double>& n_e_init, std::vector<double> n_i_init, 
        std::vector<double>& phi_init, std::vector<double> E_init, std::vector<double> x_cells_init, std::vector<double> x_nodes_init);

    // member variables / data members / fields
    std::vector<double> n_e;
    std::vector<double> n_i;
    std::vector<double> phi;
    std::vector<double> E;
    std::vector<double> x_cells;
    std::vector<double> x_nodes;
    int n_cells;
    int n_nodes;
    double dx;

    // member functions
    int poisson_solve();
    int update_E();
    double get_E_energy();

private:
    double E_energy;
    std::vector<std::vector<double>> A_inv;
    Eigen::MatrixXd A;
    std::vector<double> rho;
};

#endif // GRID_H