#include "ContainerWidget.h"



ContainerWidget::ContainerWidget() : Widget(NULL, false, false, false, false, false, square_d(-1, -1, -1, -1))
{
}

ContainerWidget::ContainerWidget(const square_d & dim) : Widget(NULL, false, false, false, false, false, dim)
{
}


ContainerWidget::~ContainerWidget()
{
}

std::string ContainerWidget::XMLName() const
{
	return staticXMLName();
}

std::string ContainerWidget::staticXMLName()
{
	return "ContainerWidget";
}
