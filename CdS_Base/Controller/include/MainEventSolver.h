#pragma once
#include <EventSolver.h>
#include <GraphicalEngine.h>
#include "MonitoringEventServer.h"

class MainEventSolver
{
public:
	MainEventSolver();
	~MainEventSolver();

	template<typename MyEvent>
	static void Associate(Action *action);
	static void Associate(const std::string &id, Action *action);
	static void SolveSdlEvent();
	static void SolveMonitoringEvent();

private:
	static EventSolver _solver;
};

class SolveWindowCreated : public Action {
public:
	SolveWindowCreated() {}
	virtual ~SolveWindowCreated() {}

	virtual void triggerAction(Event *ev) { std::cout << dynamic_cast<WindowCreatedEvent *>(ev)->getInfo() << std::endl; }
};

class SolveKeypressed : public Action {
public:
	SolveKeypressed() {}
	virtual ~SolveKeypressed() {}

	virtual void triggerAction(Event *ev) {
		KeyPressedEvent *key = dynamic_cast<KeyPressedEvent *>(ev);
		Log(LDEBUG, "Handle key event ", key->getValue());
		switch (key->getValue())
		{
		case SDLK_q:
			if(key->getDown())
				new ExitEvent();
			break;
		default:

			break;
		}
	}
};

class SolveExitEvent : public Action {
public:
	SolveExitEvent(bool *breakLoop) : _breakLoop(breakLoop) {}
	virtual ~SolveExitEvent() {}

	virtual void triggerAction(Event *ev) {
		Log(LDEBUG, "Handle Exit event");
		*_breakLoop = true;
	}

private:
	bool* _breakLoop;
};

template<typename MyEvent>
inline void MainEventSolver::Associate(Action * action)
{
	Associate(MyEvent::getId(), action);
}

