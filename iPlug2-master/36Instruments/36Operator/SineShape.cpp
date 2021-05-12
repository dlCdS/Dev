#include "SineShape.h"

SineShape::SineShape() : VirtualShapeGenerator() , _type(Shape::Type::SINE)
{
}

void SineShape::setShape(const Shape::Type& type)
{
  _type = type;
}

double SineShape::getShape(const double& t)
{
  switch (_type) {
  case Shape::Type::SINE:
    return Shape::Sine(t);
      break;
  case Shape::Type::TRIANGLE:
    return Shape::Triangle(t);
    break;
  case Shape::Type::SAWDOWN:
    return Shape::SawDown(t);
    break;
  case Shape::Type::SAWUP:
    return Shape::SawUp(t);
    break;
  case Shape::Type::SQUARE:
    return Shape::Square(t);
    break;
}
}
