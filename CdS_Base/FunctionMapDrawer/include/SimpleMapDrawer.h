#pragma once
#include "VirtualModel.h"
class SimpleMapDrawer :
	public VirtualModel
{
public:
	SimpleMapDrawer();
	~SimpleMapDrawer();

protected:
	virtual void draw();

	virtual void mainLoop();
};

