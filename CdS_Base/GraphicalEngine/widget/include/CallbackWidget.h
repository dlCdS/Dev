#pragma once
#include "Widget.h"
class CallbackWidget :
	public virtual Widget
{
	
public:
	CallbackWidget(XML::VoidCallback callback, const bool &loopback,
		XML::VoidCallback onFocus, const bool &loopbackFocus,
		XML::VoidCallback looseFocus, const bool &loopbackLooseFocus);
	CallbackWidget(XML::VoidCallback callback, const bool &loopback,
		XML::VoidCallback onFocus, const bool &loopbackFocus);
	CallbackWidget(XML::VoidCallback callback, const bool &loopback);
	CallbackWidget();
	virtual ~CallbackWidget();

	void setCallback(XML::VoidCallback callback, const bool &onLoopback = false);
	void setOnFocusCallback(XML::VoidCallback callback, const bool &onLoopback = false);
	void setFocusLooseCallback(XML::VoidCallback callback, const bool &onLoopback = false);
	const bool &loopbackAction() const;
	const bool &loopbackFocus() const;
	const bool &loopbackLooseFocus() const;
	virtual void callback(void *v);
	void onFocusCallback(std::list<Widget*> *list);
	void onFocusLooseCallback();
	virtual bool getWidgetAt(const SDL_Point & pos, std::list<Widget*> &list);
	bool getFocus() const;
	void unsetFocus();


protected:

	void onFocusDefault(void *v);
	void onLooseFocusDefault(void *v);
	virtual void noCallback(void *v);
	XML::VoidCallback _callback, _onFocus, _looseFocus;
	bool _loopbackAction, _loopbackFocus, _loopbackLooseFocus, _hasFocus;
	SDL_Point _pos;
	std::list<Widget*> *_marked;
};

