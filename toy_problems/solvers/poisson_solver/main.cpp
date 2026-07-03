#include "poisson_solver.h"
#include <iostream>
#include <vector>

using namespace Const;
using namespace std;

int main()
{
    const int ni = 21;
    const double x0 = 0;
    const double xd = 0.1;
    double dt = 1e-10;

    World world = World(ni, x0, xd, dt);
    for (int i = 0; i < ni; i++)
    {
        world.rho[i] = QE * 1e12;
    }

    solvePotentialDirect(world.dx, world.phi, world.rho);
    computeEF(world.dx, world.ef, world.phi, true);

    // generate a test electron
    double m = ME;
    double q = -QE;
    double x = 4 * world.dx;
    double v = 0;

    double li = XtoL(x, world.dx);
    double ef_p = gather(li, world.ef);
    v -= 0.5 * (q / m) * ef_p * world.dt;

    // save initial potential for PE calculation
    double phi_max = world.phi[0];
    for (int i = 1; i < ni; ++i)
        if (world.phi[i] > phi_max)
            phi_max = world.phi[i];

    // open file for particle trace
    ofstream out("trace.csv");
    if (!out)
    {
        cerr << "Failed to open trace file" << endl;
        return -1;
    }
    out << "time,x,v,KE,PE\n";
    double x_old = x;

    // particle loop
    for (int ts = 1; ts <= 4000; ++ts)
    {
        // sample mesh data at particle position
        li = XtoL(x, world.dx);
        ef_p = gather(li, world.ef);

        // integrate velocity and position
        x_old = x;
        v += (q / m) * ef_p * world.dt;
        x += v * world.dt;

        double phi_p = gather(XtoL(0.5 * (x + x_old), world.dx), world.phi);
        // phi(x(k-0.5))
        double ke = 0.5 * m * v * v / QE;            // KE in eV
        double pe = q * (phi_p - phi_max) / QE;      // PE in eV

        // write to a file
        out << ts * world.dt << " ," << x << " ," << v << " ," << ke << " ," << pe << "\n";

        if (ts == 1 || ts % 1000 == 0)  // screen output every 1000 timesteps
            cout << "ts: " << ts << ", x: " << x << ", v: " << v << ", KE: " << ke
                 << ", PE: " << pe << "\n";
    }

    out.close();
    
    // output results to a CSV file for plotting
    outputCSV(world);

    return 0;  // normal exit
}