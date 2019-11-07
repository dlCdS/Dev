#pragma once
#include <Widgetable.h>

class Modelisable
{
public:
	Modelisable(const std::string &modelName);
	virtual ~Modelisable();

	const std::string &getModelName() const;
private:
	std::string _modelName;
};


class InstanceModel :
	public Widgetable
{
public:
	virtual ~InstanceModel();
private:
	InstanceModel();
};

