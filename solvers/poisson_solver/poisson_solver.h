#ifndef POISSON_SOLVER_H
#define POISSON_SOLVER_H

#include <string>
#include <vector>
#include <random>
#include <fstream>

using namespace std;

namespace Const
{
    extern const double QE;
    extern const double EPS0;
    extern const double ME;
    extern const double AMU;
}

using dvector = vector<double>;
using Field = dvector;

struct Particle
{
    double pos;
    double vel;
    double mpw;
    Particle(double x, double v, double mpw);
};

class World
{
public:
    World(int ni, double x0, double xm, double dt);
    double getX0() const;
    double getXm() const;
    double getXc() const;

    const int ni;
    const double x0;
    const double xm;
    const double dx;
    const double dt;

    dvector x;
    dvector phi;
    dvector rho;
    dvector ef;
};

class Rnd
{
public:
    Rnd() : mt_gen(random_device{}()), rnd_dist(0.0, 1.0) {}
    double operator()() { return rnd_dist(mt_gen); }

private:
    mt19937 mt_gen;
    uniform_real_distribution<double> rnd_dist;
};

extern Rnd rnd;

class Species
{
public:
    Species(const string &name, double mass, double charge, World &world);
    size_t getNp() const;
    void advance();
    void computeNumberDensity();
    void addParticle(double pos, double vel, double mpwt);
    void loadParticlesBox(double x1, double x2, double num_den, int num_mp);

    const string name;
    const double mass;
    const double charge;
    Field den;
    vector<Particle> particles;

protected:
    World &world;
};

bool outputCSV(const World &world);
void solvePotentialDirect(double dx, dvector &phi, const dvector &rho);
void computeEF(double dx, dvector &ef, const dvector &phi, bool second_order = true);
double XtoL(double x, double dx, double x0 = 0);
double gather(double li, const dvector &field);

#endif // POISSON_SOLVER_H