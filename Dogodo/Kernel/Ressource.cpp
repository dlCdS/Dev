#include "Ressource.h"

bool Ressource::_locked = false;
ge_i Ressource::_num = 0;

Ressource::Ressource() 
{
	_locked = true;
	if (_num > 0)
		_q = new ge_i[_num];
}


Ressource::~Ressource()
{
	delete[] _q;
}

void Ressource::setSize(const ge_i & size)
{
	if(!_locked)
		_num = size;
}
