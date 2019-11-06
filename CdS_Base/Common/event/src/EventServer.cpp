#include "EventServer.h"



EventServer::EventServer()
{
}


EventServer::~EventServer()
{
	for (auto event : _event)
		delete event;
}

void EventServer::pushEvent(Event *event)
{
	_event.push_back(event);
}

Event *EventServer::popEvent()
{
	if(_event.size()>0){
		Event *event = _event.front();
		_event.pop_front();
		return event;
	}
	else return NULL;
}
