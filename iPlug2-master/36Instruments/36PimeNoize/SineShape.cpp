#include "SineShape.h"

SineShape::SineShape() : VirtualShapeGenerator()
{
}

double SineShape::getShape(const double& t)
{
  return sin(t * 2.0 * M_PI);
}
