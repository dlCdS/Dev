#include "PageWidget.h"



PageWidget::PageWidget() : Widget(NULL, 0, 0, 0, 0, 0, square_d(-1, -1, -1, -1)),
_ctarget(NULL),
_ltarget(NULL)
{
}


PageWidget::~PageWidget()
{
	freeAll();
}

void PageWidget::addWidget(Widget * widget)
{
	addWidget(widget, "Widget_" + Common::Cast<int>(_page.size()));
}

void PageWidget::addWidget(Widget * widget, const std::string name)
{
	if (_layers.size() == 0){
		Widget::addWidget(widget);
		_ltarget = &_layers[0]->_wid.front();
		_ctarget = &_child[0];
	}
	else widget->setParent(this);
	_page.insert(std::make_pair(name, widget));
}

bool PageWidget::setPage(const std::string & name)
{
	if (_page.find(name) != _page.end()) {
		*_ctarget = _page[name];
		*_ltarget = _page[name];
		sqdHasChanged(true, true);
		relHasChanged(true, true);
		return true;
	} return false;
}

void PageWidget::freeAll()
{
	Widget::freeAll();
	for (auto w : _page)
		if (w.second != *_ctarget && !w.second->isShared())
			delete w.second;
	_page.clear();
}

std::string PageWidget::XMLName() const
{
	return staticXMLName();
}

std::string PageWidget::staticXMLName()
{
	return "PageWidget";
}

void PageWidget::associate()
{
}
