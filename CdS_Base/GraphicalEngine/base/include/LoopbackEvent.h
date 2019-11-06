#pragma once

#include <EventServer.h>
#include "Defines.h"

namespace Loopback {
	class Server
	{
	public:
		Server() {}
		~Server() {}
		static void push(Event *event) { instance.pushEvent(event); }
		static Event* pop() { return instance.popEvent(); }
	private:
		static EventServer instance;
	};


}

class MouseMotionEvent : public Event {
public:
	MouseMotionEvent(const Uint32 &winId, const SDL_Point &c) : _pos(c), _winId(winId) { _id = getId(); Loopback::Server::push(this); }

	SDL_Point getPos() const { return _pos; }
	Uint32 getWinId() const { return _winId; }
	static std::string getId() { return "MouseMotionEvent"; }

private:
	SDL_Point _pos;
	Uint32 _winId;
};

class MouseClickEvent : public Event {
public:
	MouseClickEvent(const Uint8 &c, const bool &down) : _button(c), _down(down), _w(NULL) { _id = getId(); Loopback::Server::push(this); }

	Uint8 getValue() const { return _button; }
	bool getDown() const { return _down; }
	void* getWidget() { return _w; }
	void setWidget(void *w) { _w = w; }
	static std::string getId() { return "MouseClickEvent"; }

private:
	Uint8 _button;
	bool _down;
	void *_w;
};

class KeyPressedEvent : public Event {
public:
	KeyPressedEvent(const SDL_Keycode &c, const bool &down) : _key(c), _down(down) { _id = getId(); Loopback::Server::push(this); }

	SDL_Keycode getValue() const { return _key; }
	bool getDown() const { return _down; }
	static std::string getId() { return "KeyEvent"; }

private:
	SDL_Keycode _key;
	bool _down;
};

class EditTextFieldEvent : public Event {
public:
	EditTextFieldEvent(void *w) : _w(w) { _id = getId(); Loopback::Server::push(this); }

	void* getWidget() { return _w; }
	static std::string getId() { return "EditTextFieldEvent"; }

private:
	void *_w;
};

class DifferedComputationEvent : public Event {
public:
	DifferedComputationEvent(void *w) : _w(w) { _id = getId(); Loopback::Server::push(this); }

	void* getWidget() { return _w; }
	static std::string getId() { return "DifferedComputationEvent"; }

private:
	void *_w;
};

class CallbackWidgetRemovedEvent : public Event {
public:
	CallbackWidgetRemovedEvent(void *w) : _w(w) { _id = getId(); Loopback::Server::push(this); }

	void* getWidget() { return _w; }
	static std::string getId() { return "CallbackWidgetRemovedEvent"; }

private:
	void *_w;
};