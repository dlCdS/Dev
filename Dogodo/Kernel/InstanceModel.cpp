#include "InstanceModel.h"



InstanceModel::InstanceModel() : Widgetable("Instance_Model")
{
}


InstanceModel::~InstanceModel()
{
}

Modelisable::Modelisable(const std::string & modelName) : _modelName(modelName)
{
}

Modelisable::~Modelisable()
{
}

const std::string & Modelisable::getModelName() const
{
	return _modelName;
}
