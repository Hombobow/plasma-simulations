#include "grid_fast.h"
#include <vector>

using namespace std;




GridData::GridData(const int n_cells_init, const vector<double> n_e_init, const vector<double> n_i_init, const vector<double>phi_init, 
    const vector<double> E_init, const vector<double> x_cells_init, const vector<double> x_nodes_init ){
    n_cells = n_cells_init; 
    n_nodes = n_cells + 1; 
    n_e = n_e_init;  
    n_i = n_i_init; 
    phi = phi_init;
    x_cells = x_cells_init; 
    x_nodes = x_nodes_init; 
    E =E_init; // stored on nodes
    
    dx = x_cells[1] - x_cells[0]; 

    E_energy = 0.0; 



}
double GridData::get_E_energy(){
    E_energy = 0.0; 
    for (int i = 0; i < n_nodes -1; i++){
        E_energy += 0.5 * E[i] * E[i] ; 
    }

    return E_energy; 
}


void GridData::tridiagonal_factor(const std::vector<double>& a, const std::vector<double>& b,
                                  const std::vector<double>& c,
                                  std::vector<double>& cp, std::vector<double>& inv_m){
    int n = b.size();
    cp.resize(n);
    inv_m.resize(n);

    inv_m[0] = 1.0 / b[0];
    cp[0]    = c[0] * inv_m[0];

    for (int i = 1; i < n; i++){
        double m = b[i] - a[i] * cp[i - 1];  
        inv_m[i] = 1.0 / m;
        cp[i]    = c[i] * inv_m[i];
    }
}

// -------------------------------------------------------------------------
// tridiagonal_solve_factored: the RHS-DEPENDENT half of the Thomas algorithm.
//
// Given a precomputed factorization (a, cp, inv_m) from tridiagonal_factor,
// solve A x = d for any d in one forward sweep + one back-substitution.
// No divisions in the loop -- inv_m already holds the reciprocals.
//
// Writes into the caller-provided x, using dp as scratch. Both must already be
// sized to n, so this routine allocates NOTHING (safe in the hot loop).
// -------------------------------------------------------------------------
void GridData::tridiagonal_solve_factored(const std::vector<double>& a,
                                  const std::vector<double>& cp, const std::vector<double>& inv_m,
                                  const std::vector<double>& d,
                                  std::vector<double>& dp, std::vector<double>& x){
    int n = d.size();

    // Forward sweep: eliminate the sub-diagonal from the RHS.
    dp[0] = d[0] * inv_m[0];
    for (int i = 1; i < n; i++){
        dp[i] = (d[i] - a[i] * dp[i - 1]) * inv_m[i];
    }

    // Back-substitution: read off the solution bottom-to-top.
    x[n - 1] = dp[n - 1];
    for (int i = n - 2; i >= 0; i--){
        x[i] = dp[i] - cp[i] * x[i + 1];
    }
}

// -------------------------------------------------------------------------
// tridiagonal_solve: convenience one-shot solve (factor + solve) for any
// matrix you only use once. Allocates its own scratch, so it is NOT for the
// hot loop -- it exists for one-off solves (e.g. a future Dirichlet case).
// -------------------------------------------------------------------------
std::vector<double> GridData::tridiagonal_solve(const std::vector<double>& a, const std::vector<double>& b,
                                           const std::vector<double>& c, const std::vector<double>& d){
    std::vector<double> cp, inv_m;
    tridiagonal_factor(a, b, c, cp, inv_m);
    std::vector<double> dp(d.size()), x(d.size());
    tridiagonal_solve_factored(a, cp, inv_m, d, dp, x);
    return x;
}

// -------------------------------------------------------------------------
// setup_poisson: build the CONSTANT parts of the periodic Poisson solve ONCE.
//
// The Poisson matrix (the (1,-2,1) stencil plus the periodic corners) never
// changes during the simulation, so nothing here depends on the right-hand
// side. We precompute:
//   - the stencil diagonals pois_a / pois_b / pois_c and corners
//   - the Sherman-Morrison parameter gamma
//   - the modified diagonal pois_bb of the plain matrix A'
//   - pois_z = A'^{-1} u  (this is a full tridiagonal solve, done just once)
//
// After this runs, each timestep's solve is a single tridiagonal solve on the
// changing RHS instead of two. Called lazily the first time it is needed.
// -------------------------------------------------------------------------
void GridData::setup_poisson(){

    int n = n_cells;

    // The periodic Poisson stencil:  -(phi[i-1] - 2 phi[i] + phi[i+1]) = dx^2 rho.
    pois_a.assign(n, 1.0);    // sub-diagonal
    pois_b.assign(n, -2.0);   // main diagonal
    pois_c.assign(n, 1.0);    // super-diagonal
    pois_alpha = 1.0;         // bottom-left corner A[n-1][0] (wraps to phi[0])
    pois_beta  = 1.0;         // top-right   corner A[0][n-1] (wraps to phi[n-1])

    // Sherman-Morrison free parameter. -b[0] is the textbook choice; it keeps
    // the modified diagonal away from zero (good conditioning).
    pois_gamma = -pois_b[0];

    // pois_bb: main diagonal of the PLAIN tridiagonal matrix A'. Only the two
    // ends differ from pois_b; the interior is identical.
    pois_bb = pois_b;
    pois_bb[0]     = pois_b[0]     - pois_gamma;
    pois_bb[n - 1] = pois_b[n - 1] - pois_alpha * pois_beta / pois_gamma;

    // Factor the plain matrix A' ONCE. pois_cp and pois_inv_m are then reused
    // for every per-timestep solve, so each solve is just forward + back sweeps.
    tridiagonal_factor(pois_a, pois_bb, pois_c, pois_cp, pois_inv_m);

    // Size the reusable scratch buffers once, so per-timestep solves allocate nothing.
    pois_d.assign(n, 0.0);
    pois_dp.assign(n, 0.0);
    pois_y.assign(n, 0.0);
    pois_z.assign(n, 0.0);

    // u: the rank-1 correction vector, nonzero only at the two ends.
    std::vector<double> u(n, 0.0);
    u[0]     = pois_gamma;
    u[n - 1] = pois_alpha;

    // pois_z = A'^{-1} u. This does NOT depend on the RHS, so we solve it once
    // here (using the factorization we just built) and reuse it every timestep.
    tridiagonal_solve_factored(pois_a, pois_cp, pois_inv_m, u, pois_dp, pois_z);

    pois_ready = true;
}

// -------------------------------------------------------------------------
// periodic_poisson_solve: solve the periodic (cyclic tridiagonal) system for a
// given RHS r, using the precomputed operator from setup_poisson().
//
// Full Sherman-Morrison is: solve A' y = r, solve A' z = u, combine. Because
// pois_z is already known, we only do the FIRST solve here.
// -------------------------------------------------------------------------
void GridData::periodic_poisson_solve(const std::vector<double>& r, std::vector<double>& out){

    if (!pois_ready) setup_poisson();   // build the constant pieces on first use

    int n = pois_b.size();

    // The one RHS-dependent solve: A' y = r, using the precomputed factorization
    // and the member scratch buffers (no re-factoring, no divisions in the loop,
    // no allocation). Result lands in pois_y.
    tridiagonal_solve_factored(pois_a, pois_cp, pois_inv_m, r, pois_dp, pois_y);

    // Combine using the precomputed pois_z. The Sherman-Morrison v vector is
    // v = [1, 0, ..., 0, beta/gamma], so only entries 0 and n-1 appear.
    double fact = (pois_y[0] + pois_beta * pois_y[n - 1] / pois_gamma)
                / (1.0 + pois_z[0] + pois_beta * pois_z[n - 1] / pois_gamma);

    // Write straight into the caller's buffer (which will be phi).
    for (int i = 0; i < n; i++){
        out[i] = pois_y[i] - fact * pois_z[i];
    }
}

// -------------------------------------------------------------------------
// poisson_solve: solve  -phi'' = rho  on the PERIODIC grid for the potential.
//
// Discretized with the 3-point stencil on cell centers:
//     -(phi[i-1] - 2 phi[i] + phi[i+1]) / dx^2 = rho[i]
// Multiplying by -dx^2 gives the tridiagonal coefficients (1, -2, 1) and
// right-hand side d[i] = -dx^2 * rho[i]. Periodic BCs add the wrap corners
// alpha = beta = 1, so we must use cyclic_solve, NOT thomas_solve directly.
//
// Two subtleties because periodic Poisson is SINGULAR (the matrix has the
// constant vector in its null space):
//   (1) A solution only exists if the net charge is zero: sum(rho) = 0. We
//       enforce this by subtracting the mean of rho before solving.
//   (2) phi is only determined up to an additive constant. We fix that "gauge"
//       by subtracting the mean of phi afterward, so mean(phi) = 0. (E = -dphi
//       /dx is unaffected by the constant, so this choice is physically free.)
// -------------------------------------------------------------------------
int GridData::poisson_solve(){

    if (!pois_ready) setup_poisson();   // ensures the scratch buffers (incl. pois_d) are sized

    int n = n_cells;

    // Build only the RHS here (the one thing that changes each timestep), writing
    // into the reusable member buffer pois_d. The matrix itself lives in the
    // precomputed members (setup_poisson()).
    //
    // rho = n_i - n_e. We subtract its mean so the net charge is zero, which is
    // required for a periodic solution to exist (see the singularity note).
    double rho_mean = 0.0;
    for (int i = 0; i < n; i++) rho_mean += (n_i[i] - n_e[i]);
    rho_mean /= n;

    for (int i = 0; i < n; i++){
        pois_d[i] = -(dx * dx) * ((n_i[i] - n_e[i]) - rho_mean);
    }

    // Periodic solve using the precomputed operator, written straight into phi.
    periodic_poisson_solve(pois_d, phi);

    // Gauge fix: phi is only defined up to a constant, so pin mean(phi) = 0.
    double phi_mean = 0.0;
    for (int i = 0; i < n; i++) phi_mean += phi[i];
    phi_mean /= n;
    for (int i = 0; i < n; i++) phi[i] -= phi_mean;

    return 1;
}


int GridData::update_E(){
    for (int i = 1; i < n_nodes -1 ; i++){
        E[i] = -(phi[i] - phi[i - 1])/dx ;
    }
    E[0] = -(phi[0] - phi[n_cells - 1 ])/dx ; 
    E[n_nodes - 1] = -(phi[0] - phi[n_cells - 1])/dx ; 
    return 1; 
}



