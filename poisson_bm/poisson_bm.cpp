#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace Const
{
    const double QE = 1.602176565e-19;    // C, electron charge
    const double EPS0 = 8.85418782e-12;   // C/V/m, vac. permittivity
    const double ME = 9.10938215e-31;     // kg , electron mass
    const double AMU = 1.66053906660e-27; // kg, atomic mass unit
}

using namespace std;
using namespace Const;
using dvector = vector<double>;

std::vector<int> logspace_int(double start, double end, int num)
{
    std::vector<int> result(num);
    double log_start = std::log10(start);
    double log_end = std::log10(end);
    double step = (log_end - log_start) / (num - 1);
    for (int i = 0; i < num; i++)
    {
        double val = std::pow(10.0, log_start + i * step);
        result[i] = static_cast<int>(std::round(val));
    }
    return result;
}

void solvePotentialDirect(double dx, dvector &phi, dvector &f)
{
    int ni = static_cast<int>(phi.size());
    if (ni < 2)
    {
        return;
    }

    dvector a(ni);
    dvector b(ni);
    dvector c(ni);
    dvector d(ni);

    double dx2 = dx * dx;
    for (int i = 0; i < ni; ++i)
    {
        if (i == 0)
        {
            b[i] = 1.0;
            c[i] = 0.0;
            d[i] = 0.0;
        }
        else if (i == ni - 1)
        {
            b[i] = 1.0;
            c[i] = 0.0;
            d[i] = 1.0;
        }
        else
        {
            a[i] = 1.0 / dx2;
            b[i] = -2.0 / dx2;
            c[i] = 1.0 / dx2;
            d[i] = f[i];
        }
    }

    d[0] = d[0] / b[0];
    for (int i = 1; i < ni; ++i)
    {
        if (i < ni - 1)
        {
            c[i] = c[i] / (b[i] - a[i] * c[i - 1]);
        }
        d[i] = (d[i] - a[i] * d[i - 1]) / (b[i] - a[i] * c[i - 1]);
    }

    phi[ni - 1] = d[ni - 1];
    for (int i = ni - 2; i >= 0; --i)
    {
        phi[i] = d[i] - c[i] * phi[i + 1];
    }
}

void writePotentialToCsv(const std::string &filename, const dvector &x, const dvector &phi)
{
    std::ofstream out(filename);
    if (!out)
    {
        std::cerr << "Could not open " << filename << " for writing." << std::endl;
        return;
    }

    out << "x,phi\n";
    for (size_t i = 0; i < x.size(); ++i)
    {
        out << x[i] << "," << phi[i] << "\n";
    }
}

void results()
{
    const double x0 = 0.0;
    const double xm = 1.0;
    const int ni = 33;
    const double dx = (xm - x0) / (ni - 1);

    dvector x(ni);
    dvector phi(ni);
    dvector f(ni);

    for (int i = 0; i < ni; ++i)
    {
        x[i] = x0 + i * dx;
        f[i] = sin(x[i]) + cos(x[i]);
    }

    solvePotentialDirect(dx, phi, f);
    writePotentialToCsv("phi_solution.csv", x, phi);
    std::cout << "Wrote phi solution to phi_solution.csv" << std::endl;
}

double analyticPotential(double x)
{
    return -sin(x) - cos(x) + (sin(1) + cos(1)) * x + 1;
}

double computeAverageAbsoluteError(int ni)
{
    const double x0 = 0.0;
    const double xm = 1.0;
    const double dx = (xm - x0) / (ni - 1);

    dvector x(ni);
    dvector phi(ni);
    dvector f(ni);

    for (int i = 0; i < ni; ++i)
    {
        x[i] = x0 + i * dx;
        f[i] = sin(x[i]) + cos(x[i]);
    }

    solvePotentialDirect(dx, phi, f);

    double sum = 0.0;
    for (int i = 0; i < ni; ++i)
    {
        double phi_a = analyticPotential(x[i]);
        sum += abs(phi[i] - phi_a);
    }

    return sum / ni;
}

void computeErrorCsv()
{
    const double x0 = 0.0;
    const double xm = 1.0;
    const vector<int> gridSizes = logspace_int(11, 1001, 8);

    const string filename = "error_vs_dx.csv";

    std::ofstream out(filename);
    if (!out)
    {
        cerr << "Could not open " << filename << " for writing." << std::endl;
        return;
    }

    out << "ni,dx,avg_abs_error\n";
    for (int ni : gridSizes)
    {
        double dx = (xm - x0) / (ni - 1);
        double error = computeAverageAbsoluteError(ni);
        out << ni << "," << dx << "," << error << "\n";
    }

    cout << "Wrote error data to " << filename << endl;
}

int main()
{
    results();
    computeErrorCsv();
    return 0;
}