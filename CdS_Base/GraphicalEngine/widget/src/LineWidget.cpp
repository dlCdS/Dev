#include "LineWidget.h"



LineWidget::LineWidget()
{
}


LineWidget::~LineWidget()
{
}

std::string LineWidget::XMLName() const
{
	return staticXMLName();
}

std::string LineWidget::staticXMLName()
{
	return "LineWidget";
}
