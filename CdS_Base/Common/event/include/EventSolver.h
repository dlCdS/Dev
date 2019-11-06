#pragma once

#include <unordered_map>
#include "Action.h"

class EventSolver
{
public:
	EventSolver();
	~EventSolver();

	void solveEvent(Event *event);
	void associate(const std::string &id, Action *action);

private:
	std::unordered_map<std::string, Action*> _solver;
};

