#include "ListWidget.h"


ListWidget::ListWidget() : Widget(NULL, 1, 0, 0, 0, 1, square_d(-1, -1, -1, -1))
{
}


ListWidget::~ListWidget()
{
}

std::string ListWidget::XMLName() const
{
	return staticXMLName();
}

std::string ListWidget::staticXMLName()
{
	return "listWidget";
}
