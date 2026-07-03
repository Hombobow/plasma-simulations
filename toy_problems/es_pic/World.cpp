#include "World.h"

// constructor
World::World(int ni, int nj, int nk) : ni{ni}, nj{nj}, nk{nk}, nn{ni, nj, nk} {}

// sets mesh extents
void World::setExtents(double x1, double y1, double z1, double x2, double y2, double z2)
{
}