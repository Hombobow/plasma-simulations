#include "grid.h"
#include <vector>

using namespace std;

GridData::GridData(const int& n_cells_init, const std::vector<double>& n_e_init, std::vector<double> n_i_init, 
    std::vector<double>& phi_init, std::vector<double> E_init, std::vector<double> x_cells_init, std::vector<double> x_nodes_init) {
    n_cells = n_cells_init;
    n_nodes = n_cells_init + 1;
    n_e = n_e_init;
    n_i = n_i_init;
    phi = phi_init;
    E = E_init;
    x_cells = x_cells_init;
    x_nodes = x_nodes_init;

    rho.assign(n_cells, 0.0);
    dx = x_cells[1] - x_cells[0];
    E_energy = 0.0; 
    
    A = Eigen::MatrixXd(n_cells, n_cells);
    A.setZero();
    A_inv.assign(n_cells, std::vector<double>(n_cells, 0.0));

    for (int i = 1; i < n_cells; i++) {
        A(i, i) = -2.0;
        A(i, (i + 1) % n_cells) = 1.0; // super diagonal wraps to column 0 when i = n - 1
        A(i, (i - 1 + n_cells) % n_cells) = 1.0; // sub diagonal wraps to column n - 1 when i = 0
    }
    
    A(0, 0) = 1.0; // the laplacian matrix is singular, enforces phi[0] = 0
    A = A.inverse();

    // Store dx^2 * A^{-1} so poisson_solve can do phi = -A_inv * rho
    // (previously multiplied the still-zero A_inv, so E was always 0).
    for (int i = 0; i < n_cells; i++) {
        for (int j = 0; j < n_cells; j++) {
            A_inv[i][j] = A(i, j) * dx * dx;
        }
    }
}

int GridData::poisson_solve() {
    phi[0] = 0; 
    rho[0] = 0; // to fix non-singularity of matrix/solutio
    for (int i = 1; i < n_cells; i++) rho[i] = n_i[i] - n_e[i];

    for (int i = 0; i < n_cells; i++) {
        phi[i] = 0;
        for (int j = 0; j < n_cells; j++) {
            phi[i] -= A_inv[i][j] * rho[j];
        }
    }

    return 1;
}

int GridData::update_E() {
    for (int i = 1; i < n_nodes - 1; i++) {
        E[i] = - (phi[i] - phi[i - 1]) / dx;
    }
    E[0] = - (phi[0] - phi[n_cells - 1]) / dx;
    E[n_nodes - 1] = -(phi[0] - phi[n_cells - 1])/dx; 

    return 1;
}

double GridData::get_E_energy() {
    E_energy = 0.0;
    for (int i = 0; i < n_nodes - 1; i++) {
        E_energy += 0.5 * E[i] * E[i] * dx;
    }
    
    return E_energy;
}