#pragma once
#include "../36Common/VirtualShapeGenerator.h"
#define _USE_MATH_DEFINES
#include <math.h>

class SineShape :
    public VirtualShapeGenerator
{
public:
  SineShape();

protected:
  virtual double getShape(const double& t);
};

