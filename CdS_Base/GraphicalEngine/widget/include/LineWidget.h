#pragma once
#include "Widget.h"
class LineWidget :
	public Widget
{
public:
	LineWidget();
	~LineWidget();

	virtual std::string XMLName() const;
	static std::string staticXMLName();

};

