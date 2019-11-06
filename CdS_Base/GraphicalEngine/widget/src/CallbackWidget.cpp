#include "CallbackWidget.h"




CallbackWidget::CallbackWidget(XML::VoidCallback callback, const bool & loopback, 
	XML::VoidCallback onFocus, const bool & loopbackFocus, 
	XML::VoidCallback looseFocus, const bool & loopbackLooseFocus) :
	Widget(),
	_callback(callback),
	_loopbackAction(loopback),
	_onFocus(onFocus),
	_loopbackFocus(loopbackFocus),
	_looseFocus(looseFocus),
	_loopbackLooseFocus(loopbackLooseFocus),
	_hasFocus(false),
	_marked(NULL)
{
}

CallbackWidget::CallbackWidget(XML::VoidCallback callback, const bool &loopback, XML::VoidCallback onFocus, const bool &loopbackFocus) :
	Widget(),
	_callback(callback),
	_loopbackAction(loopback),
	_onFocus(onFocus),
	_loopbackFocus(loopbackFocus),
	_looseFocus(VoidedCallbackFunction(CallbackWidget, onLooseFocusDefault)),
	_loopbackLooseFocus(true),
	_hasFocus(false),
	_marked(NULL)
{
}

CallbackWidget::CallbackWidget(XML::VoidCallback callback, const bool & loopback) :
	Widget(),
	_callback(callback),
	_loopbackAction(loopback),
	_onFocus(VoidedCallbackFunction(CallbackWidget, onFocusDefault)),
	_loopbackFocus(true),
	_looseFocus(VoidedCallbackFunction(CallbackWidget, onLooseFocusDefault)),
	_loopbackLooseFocus(true),
	_hasFocus(false),
	_marked(NULL)
{
}

CallbackWidget::CallbackWidget() :
	Widget(),
	_callback(VoidedCallbackFunction(CallbackWidget, noCallback)),
	_loopbackAction(true),
	_onFocus(VoidedCallbackFunction(CallbackWidget, onFocusDefault)),
	_loopbackFocus(true),
	_looseFocus(VoidedCallbackFunction(CallbackWidget, onLooseFocusDefault)),
	_loopbackLooseFocus(true),
	_hasFocus(false),
	_marked(NULL)
{
}

CallbackWidget::~CallbackWidget()
{
	if (_marked != NULL){
		_marked->remove(this);
		new CallbackWidgetRemovedEvent(this);
	}
}

void CallbackWidget::setCallback(XML::VoidCallback callback, const bool & onLoopback)
{
	_loopbackAction = onLoopback;
	_callback = callback;
}

void CallbackWidget::setOnFocusCallback(XML::VoidCallback callback, const bool & onLoopback)
{
	_loopbackFocus = onLoopback;
	_onFocus = callback;
}

void CallbackWidget::setFocusLooseCallback(XML::VoidCallback callback, const bool & onLoopback)
{
	_loopbackLooseFocus = onLoopback;
	_looseFocus = callback;
}

const bool & CallbackWidget::loopbackAction() const
{
	return _loopbackAction;
}

const bool & CallbackWidget::loopbackFocus() const
{
	return _loopbackFocus;
}

const bool & CallbackWidget::loopbackLooseFocus() const
{
	return _loopbackLooseFocus;
}

void CallbackWidget::callback(void *v)
{
	_callback(v);
}

void CallbackWidget::onFocusCallback(std::list<Widget*> *list)
{
	_marked = list;
	_hasFocus = true;
	_onFocus(NULL);
}

void CallbackWidget::onFocusLooseCallback()
{
	_marked = NULL;
	_hasFocus = false;
	_looseFocus(NULL);
}

bool CallbackWidget::getWidgetAt(const SDL_Point & pos, std::list<Widget*>& list)
{
	if (SDL_PointInRect(&pos, &_abs)) {
		if (_clickable) {
			list.push_back(this);
			_pos = pos;
		}
		for (auto l = _layers.rbegin(); l != _layers.rend(); ++l)
			for (auto w = (*l)->_wid.begin(); w != (*l)->_wid.end(); ++w)
				if ((*w)->getWidgetAt(pos, list))
					return true;
		return true;
	}
	return false;
}

bool CallbackWidget::getFocus() const
{
	return _hasFocus;
}

void CallbackWidget::unsetFocus()
{
	_hasFocus = false;
}

void CallbackWidget::onFocusDefault(void * v)
{
	setColor(200, 200, 200);
}

void CallbackWidget::onLooseFocusDefault(void * v)
{
	setColor();
}

void CallbackWidget::noCallback(void * v)
{
	Log(LWARNING, "No callback defined on button");
}
