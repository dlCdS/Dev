#pragma once
#include <EventServer.h>

class KernelEventServer
{
public:
	KernelEventServer();
	~KernelEventServer();

	static void push(Event *event) { instance.pushEvent(event); }
	static Event* pop() { return instance.popEvent(); }
private:
	static EventServer instance;
};

