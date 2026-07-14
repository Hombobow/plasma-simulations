#include "World.h"

// constructor
World::World(int ni, int nj, int nk) : ni{ni}, nj{nj}, nk{nk}, nn{ni, nj, nk} {}

// sets mesh extents and computes cell spacing
void World::setExtents(double x1, double y1, double z1, double x2, double y2, double z2)
{
    // set orgin (xmin)
    x0[0], x0[1], x0[2] = x1, y1, z1;

    // diagonally-opposite corner (xmax)
    xm[0], xm[1], xm[2] = x2, y2, z2;

    // compute spacing by dividing length by the number of cells
    for (int i = 0; i < 3; i++)
    {
        dh[i] = (xm[i] - x0[i]) / (nn[i] - 1);
    }

    // compute centroid
    for (int i = 0; i < 3; i++)
    {
        xc[i] = 0.5 * (x0[i] + xm[i]);
    }
}
