#pragma once

#include <string>
#include "Logger.h"

class Event
{
public:
	Event();
	virtual ~Event();

	std::string getId() const;
	void forward(const bool &doForward);
	const bool &forwarded() const;

protected:
	std::string _id;
	bool _forwarded;
};

