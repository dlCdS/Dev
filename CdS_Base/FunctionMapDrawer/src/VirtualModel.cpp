#include "VirtualModel.h"

VirtualModel::VirtualModel()
{
}

VirtualModel::~VirtualModel()
{
}

void VirtualModel::setColourWidget(SetColourWidget* scw)
{
	_scw = scw;
	initWidget(_scw);
}

void VirtualModel::setSize(const ge_pi& size, const ge_i& zoom)
{
	_size = size;
	_zoom = zoom;
}

void VirtualModel::cycle()
{
	mainLoop();
	draw();
}

void VirtualModel::initWidget(SetColourWidget* widget)
{
	Log(LINFO, "Init widget");
	widget->setSize({ _size.w * _zoom, _size.h * _zoom });
	widget->generateSurface();
	widget->setupForRect();
}

