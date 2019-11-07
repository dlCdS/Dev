#pragma once
#include <Common.h>
#include <unordered_map>

class Ressource
{
public:
	Ressource();
	~Ressource();

	static void setSize(const ge_i &size);

protected:
	ge_i *_q;
	static bool _locked;
	static ge_i _num;
};

