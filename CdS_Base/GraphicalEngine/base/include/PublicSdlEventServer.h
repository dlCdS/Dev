#pragma once
#include <EventServer.h>
#include "Defines.h"

class PublicSdlEventServer :
	public EventServer
{
public:
	PublicSdlEventServer();
	~PublicSdlEventServer();

	static void push(Event *event) { instance.pushEvent(event); }
	static Event* pop() { return instance.popEvent(); }
private:
	static EventServer instance;
};

class MouseScrollEvent : public Event {
public:
	MouseScrollEvent(const unsigned int &dir) : _dir(dir) { _id = getId(); PublicSdlEventServer::push(this); }

	unsigned int getValue() const { return _dir; }
	static std::string getId() { return "MouseScrollEvent"; }

private:
	unsigned int _dir;
};



class WindowCreatedEvent : public Event {
public:
	WindowCreatedEvent(const std::string &info) : _info(info) { _id = getId();  PublicSdlEventServer::push(this); }

	std::string getInfo() const { return _info; }
	static std::string getId() { return "WindowCreatedEvent"; }
private:
	std::string _info;
};
