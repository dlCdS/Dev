#include "Event.h"


Event::Event() : 
	_forwarded(false)
{
}


Event::~Event()
{
}

std::string Event::getId() const
{
	return _id;
}

void Event::forward(const bool &doForward)
{
	_forwarded = doForward;
}

const bool & Event::forwarded() const
{
	return _forwarded;
}

