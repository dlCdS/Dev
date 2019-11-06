#pragma once
#include "Widget.h"
class PageWidget :
	public Widget
{
public:
	PageWidget();
	virtual ~PageWidget();

	virtual void addWidget(Widget *widget);
	void addWidget(Widget *widget, const std::string name);
	bool setPage(const std::string &name);
	void removePage(const std::string &name);
	virtual void freeAll();


	virtual std::string XMLName() const;
	static std::string staticXMLName();

protected:

	virtual void associate();
	Widget **_ctarget, **_ltarget;
	std::unordered_map<std::string, Widget *> _page;
};

