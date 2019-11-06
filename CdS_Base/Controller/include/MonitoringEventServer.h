#pragma once
#include <EventServer.h>
class MonitoringEventServer 
{
public:
	MonitoringEventServer();
	~MonitoringEventServer();

	static void push(Event *event) { instance.pushEvent(event); }
	static Event* pop() { return instance.popEvent(); }
private:
	static EventServer instance;
};


class ExitEvent : public Event {
public:
	ExitEvent()  { _id = getId(); MonitoringEventServer::push(this); }

	static std::string getId() { return "ExitEvent"; }

private:
};

