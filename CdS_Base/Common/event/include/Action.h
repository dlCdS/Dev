#pragma once
#include "Event.h"

class Action
{
public:
	Action();
	virtual ~Action();

	virtual void triggerAction(Event *event) = 0;
};

