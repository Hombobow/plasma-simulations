#ifndef _WORLD_H
#define _WORLD_H

class World
{
public:
    World(int ni, int nj, int nk); // constructor

    // sets the mesh span, also recomputes cell spanning
    void setExtents(double x1, double y1, double z1, double x2, double y2, double z2);

    const int nn[3];      // number of nodes in x, y, z
    const int ni, nj, nk; // number of nodes in individual variables

protected:
    double x0[3]; // mesh orgin
    double dh[3]; // cell spacing x, y, z
    double xm[3]; // mesh max bound
    double xc[3]; // domain centroid
};

#endif