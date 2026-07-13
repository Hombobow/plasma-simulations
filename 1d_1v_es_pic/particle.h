#ifndef PARTICLE_H
#define PARTICLE_H

// header file for particle class

class Particle{
    private: 

        double m; 
        double q; 

        
    public: 
        double x; 
        double vx; 
        double w ; 
        // constructor
        Particle(double x_init = 0.0, double vx_init = 0.0 , double m = 1.0, double q = -1.0 , double w = 1.0);

        double get_x();
        double get_vx(); 

        double get_m(); 
        double get_q(); 
        
        void update_x(double dt, double domain_L); 
        void update_vx(double dt, double F); 

};


#endif