#pragma once
#include "Widget.h"
class FitLineWidget :
	public Widget
{
public:
	FitLineWidget();
	FitLineWidget(Widget *parent);
	~FitLineWidget();

	virtual void addWidget(Widget *widget);

	virtual std::string XMLName() const;
	static std::string staticXMLName();

protected:
	virtual void postComputeRelative(const ge_pi &pen);
};

