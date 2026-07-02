#include "poisson_solver.h"
#include <iostream>
#include <vector>

using namespace Const;
using namespace std;

int main()
{
    // load particle
    double m = ME;
    double q = -QE;
    double x = 0;
    double v = 0;

    // simulation parameters
    double dt = 1e-9;
    double E = -100;

    // open particle trace file
    ofstream out("trace.csv");
    if (!out)
    {
        cerr << "Failed to open file" << endl;
        return -1;
    }
    out << "t,x,v\n";

    v -= 0.5 * (q / m) * E * dt;

    // particle loop
    for (int it = 0; it < 10; it++)
    {
        // write trace data
        out << it * dt << " ," << x << " ," << v << " \n";

        // advance velocity and position
        x += v * dt;
        v += (q / m) * E * dt;
    }

    return 0;
}