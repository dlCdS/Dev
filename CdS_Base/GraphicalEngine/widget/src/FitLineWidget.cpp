#include "FitLineWidget.h"



FitLineWidget::FitLineWidget() : Widget()
{
}

FitLineWidget::FitLineWidget(Widget *parent) : Widget(parent)
{
}


FitLineWidget::~FitLineWidget()
{
}

void FitLineWidget::addWidget(Widget * widget)
{
	Widget::addWidget(widget);
	widget->setRelativeProportion(square_d(-1, -1, -1, -1));
}

void FitLineWidget::postComputeRelative(const ge_pi & pen)
{
	commonComputeRelative(pen);
	int heigh = 0;
	for (auto l : _layers)
		for (auto w : l->_wid)
			heigh = max(heigh, w->getRelativeRect().h);
	_rel.h = heigh;
}

std::string FitLineWidget::XMLName() const
{
	return staticXMLName();
}

std::string FitLineWidget::staticXMLName()
{
	return "fitLineWidget";
}

