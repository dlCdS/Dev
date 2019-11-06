#include "EventSolver.h"



EventSolver::EventSolver()
{
}


EventSolver::~EventSolver()
{
	for (auto pair : _solver)
		delete pair.second;
}

void EventSolver::solveEvent(Event * event)
{
	if(event != NULL) {
		event->forward(false);
		auto a = _solver.find(event->getId());
		if (a != _solver.end()) {
			a->second->triggerAction(event);
			if(!event->forwarded())
				delete event;
		}
		else Log(LINFO, "No action for event ", event->getId());
	} 
}

void EventSolver::associate(const std::string & id, Action * action)
{
	_solver.insert(std::make_pair(id, action));
}
