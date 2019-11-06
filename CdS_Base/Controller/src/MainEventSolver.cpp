#include "MainEventSolver.h"

EventSolver MainEventSolver::_solver = EventSolver();

MainEventSolver::MainEventSolver()
{
}


MainEventSolver::~MainEventSolver()
{
}

void MainEventSolver::Associate(const std::string & id, Action * action)
{
	_solver.associate(id, action);
}

void MainEventSolver::SolveSdlEvent()
{
	Event *ev = NULL;
	do {
		ev = GE::popEvent();
		_solver.solveEvent(ev);
	} while (ev != NULL);
}

void MainEventSolver::SolveMonitoringEvent()
{
	Event *ev = NULL;
	do {
		ev = MonitoringEventServer::pop();
		_solver.solveEvent(ev);
	} while (ev != NULL);
}
