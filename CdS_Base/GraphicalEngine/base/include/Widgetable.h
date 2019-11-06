#pragma once
#include <XMLCompliant.h>
#include "Window.h"

#define MAIN_PAGE_NAME "mainPageName"


class Widgetable :
	public XML::Compliant
{
	struct BuildWidget {
		BuildWidget(const std::string &widpos, Widget *w) : _widpos(widpos), _w(w) {}
		std::string _widpos;
		Widget *_w;
	};

public:
	Widgetable(const std::string &name);
	Widgetable(const std::string &name, const bool &shared);
	virtual ~Widgetable();

	Widget *getDetachableContainer();
	Widget *getContainer();
	void setAsPage(Widgetable *prev);
	void build();

protected:
	void WAddCallback(const std::string &name, XML::VoidCallback func);
	void WAssociateWidget(const std::string &widpos, Widget* wid);
	void WAssociateBeacon(const std::string &item, const std::string &widpos);
	void WAssociateField(const std::string &item, const std::string &widpos, XML::Field *field);
	void WAssociateField(const std::string &item, const std::string &widpos, XML::Field *field, XML::VoidCallback forwardChange);
	void variableChanged(void *var);
	void WAddPage(const std::string &name, const std::string &widpos, Widgetable *widget);

	void backMain();


	PageWidget _container, *_target;
	ListWidget *_main;

private:


	void pageBack(void *v);
	void selectWidget(void *v);

	bool createField(const std::string &item, Widget *dest);
	bool createPage(const std::string &item, Widget *dest);

	void resetContainer();
	Widget *getFromName(const std::string &name, std::unordered_map<std::string, Widget*> &_fromName);
	void buildWidget(std::unordered_map<std::string, Widget*> &_fromName);
	void addUpdateVar(void *var, Widget*w);
	virtual void associate() = 0;
	bool _editable, _displayHeader, _isShared, _building;
	std::string _header;

	std::unordered_map<std::string, XML::VoidCallback> *_callback;
	std::vector<BuildWidget> *_display;
	std::vector<std::string> *_order;
	std::unordered_map<std::string, std::string> *_associate;
	std::unordered_map<std::string, XML::Field *> *_displayField;
	std::unordered_map<std::string, XML::VoidCallback> *_displayForward;
	std::unordered_map<void *, Widget *> *_updateVar;
	std::unordered_map<std::string, Widgetable *> *_widgetablePage;


	void deleteContainer(void *v);
	virtual void clearAssociation();
	Widgetable *_prevPage;
};

