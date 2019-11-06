#pragma once
#include "Widget.h"
class ContainerWidget :
	public Widget
{
public:
	ContainerWidget();
	ContainerWidget(const square_d &dim);
	~ContainerWidget();


	std::string XMLName() const;
	static std::string staticXMLName();

};

