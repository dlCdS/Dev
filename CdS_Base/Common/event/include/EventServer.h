#pragma once

#include "Event.h"
#include <list>

class EventServer
{
public:
	EventServer();
	~EventServer();


	void pushEvent(Event *event);
	Event *popEvent();

private:
	std::list<Event*> _event;
};

