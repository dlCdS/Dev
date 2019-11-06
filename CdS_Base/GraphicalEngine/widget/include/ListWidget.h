#pragma once
#include "Widget.h"
class ListWidget :
	public Widget
{
public:
	ListWidget();
	~ListWidget();

	std::string XMLName() const;
	static std::string staticXMLName();

};

