#include "particle.h"
#include <cmath>

// Parameter order must match particle.h: (x, vx, w, m, q).
// Previously (x, vx, m, q, w) swapped defaults so q became +1 → anti-shielding / energy blowup.
Particle::Particle(double x_init, double vx_init, double w_init, double m_init, double q_init) {
    x = x_init;
    vx = vx_init;
    w = w_init;
    m = m_init;
    q = q_init;
}

double Particle::get_x(){
    return x; 
}

double Particle::get_vx(){
    return vx; 
}

double Particle::get_m(){
    return m; 
}

double Particle::get_q(){
    return q; 
}

void Particle::update_x(double dt, double domain_L) {
    x = std::fmod((x + vx * dt), domain_L);
    if (x < 0) {
        x += domain_L; 
    }
}

void Particle::update_vx(double dt, double F) {
    vx = vx + F * dt;
}