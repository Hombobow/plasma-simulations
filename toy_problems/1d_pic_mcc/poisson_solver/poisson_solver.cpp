#include <vector>   // Provides the std::vector container for dynamic arrays
#include <iostream> // Provides std::cout, std::cin, std::cerr for console I/O
#include <fstream>  // Provides std::ifstream and std::ofstream for file input/output

namespace Const
{
    const double QE = 1.602176565e-19;  // C, electron charge
    const double EPS0 = 8.85418782e-12; // C/V/m, vac. permittivity
    const double ME = 9.10938215e-31;   // kg , electron mass
};

using namespace std;
using namespace Const;
using dvector = vector<double>;

// function prototype
bool outputCSV(double x0, double dx, const dvector &phi, const dvector &rho, const dvector &ef);

// main
int main()
{
    const int ni = 21;                // number of nodes
    const double x0 = 0;              // mesh orgin
    const double xm = 0.1;            // opposite end
    double dx = (xm - x0) / (ni - 1); // node spacing

    dvector phi(ni);
    dvector rho(ni, QE * 1e12);
    dvector ef(ni);

    // output to a CSV file for plotting
    outputCSV(x0, dx, phi, rho, ef);

    return 0; // exit
};

// outputs the given fields to a CSV file, returns true if ok
bool outputCSV(double x0, double dx, const dvector &phi, const dvector &rho, const dvector &ef)
{
    ofstream out("results.csv"); // open file for writing
    if (!out)
    {
        cerr << "Could not open output file!" << endl;
        return false;
    }

    out << "x, phi, rho, ef\n"; // write header
    for (int i = 0; i < phi.size(); i++)
    {
        out << x0 + i * dx;
        out << ", " << phi[i] << ", " << rho[i] << ", " << ef[i] << "\n";
    }

    return true; // file closed automatically when "out" goes out of scope
}