#include "poisson_solver.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <random>

namespace Const
{
    const double QE = 1.602176565e-19;    // C, electron charge
    const double EPS0 = 8.85418782e-12;   // C/V/m, vac. permittivity
    const double ME = 9.10938215e-31;     // kg , electron mass
    const double AMU = 1.66053906660e-27; // kg, atomic mass unit
}

using namespace Const;
using namespace std;

Particle::Particle(double x, double v, double mpw) : pos(x), vel(v), mpw(mpw) {}

World::World(int ni, double x0, double xm, double dt)
    : ni{ni}, x0{x0}, xm{xm}, dx{(xm - x0) / (ni - 1)}, dt{dt}, x(ni), phi(ni), rho(ni), ef(ni)
{
    for (int i = 0; i < ni; ++i)
    {
        x[i] = x0 + i * dx;
    }
}

double World::getX0() const { return x0; }
double World::getXm() const { return xm; }
double World::getXc() const { return 0.5 * (x0 + xm); }

Species::Species(const string &name, double mass, double charge, World &world)
    : name(name), mass(mass), charge(charge), den(world.ni, 0.0), world(world)
{
}

size_t Species::getNp() const
{
    return particles.size();
}

void Species::loadParticlesBox(double x1, double x2, double num_den, int num_mp)
{
    double interval_length = x2 - x1;
    double num_real = num_den * interval_length;
    double mpw = num_real / num_mp;

    particles.reserve(num_mp);

    for (int p = 0; p < num_mp; ++p)
    {
        double pos = x1 + (x2 - x1) * rnd();
        double vel = 0.0;
        addParticle(pos, vel, mpw);
    }
}

void Species::addParticle(double pos, double vel, double mpwt)
{
    particles.emplace_back(pos, vel, mpwt);
}

void Species::computeNumberDensity()
{
    fill(den.begin(), den.end(), 0.0);
}

void Species::advance()
{
    // TODO: move particles using world.ef[]
}

bool outputCSV(const World &world)
{
    ofstream out("results.csv");
    if (!out)
    {
        cerr << "Could not open output file!" << endl;
        return false;
    }

    out << "x, phi, rho, ef\n";
    for (int i = 0; i < world.ni; ++i)
    {
        out << world.x[i] << ", " << world.phi[i] << ", " << world.rho[i] << ", " << world.ef[i] << "\n";
    }

    return true;
}

void solvePotentialDirect(double dx, dvector &phi, const dvector &rho)
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
        if (i == 0 || i == ni - 1)
        {
            b[i] = 1.0;
            c[i] = 0.0;
            d[i] = 0.0;
        }
        else
        {
            a[i] = 1.0 / dx2;
            b[i] = -2.0 / dx2;
            c[i] = 1.0 / dx2;
            d[i] = -rho[i] / EPS0;
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

void computeEF(double dx, dvector &ef, const dvector &phi, bool second_order)
{
    int ni = phi.size();

    // central difference on internal nodes
    for (int i = 1; i < ni - 1; ++i)
        ef[i] = -(phi[i + 1] - phi[i - 1]) / (2 * dx);

    // one-sided first or second order difference on the boundaries
    if (second_order)
    {
        ef[0] = (3 * phi[0] - 4 * phi[1] + phi[2]) / (2 * dx);
        ef[ni - 1] = (-phi[ni - 3] + 4 * phi[ni - 2] - 3 * phi[ni - 1]) / (2 * dx);
    }
    else // first order
    {
        ef[0] = (phi[0] - phi[1]) / dx;
        ef[ni - 1] = (phi[ni - 2] - phi[ni - 1]) / dx;
    }
}

double XtoL(double x, double dx, double x0)
{
    return (x - x0) / dx;
}

double gather(double li, const dvector &field)
{
    int i = (int)li;
    double di = li - i;
    return field[i] * (1 - di) + field[i + 1] * di;
}

Rnd rnd;
