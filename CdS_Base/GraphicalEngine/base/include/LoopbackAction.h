#pragma once
#include "LoopbackEvent.h"
#include "Window.h"
#include <EventSolver.h>
#include "PublicSdlEventServer.h"

namespace Loopback {
	class Solver
	{
	public:
		Solver() {}
		~Solver() {}



		static void Associate(const std::string &id, Action *action) {
			_solver.associate(id, action);
		}

		static void SolveEvent() {
			Event *ev = NULL;
			do {
				ev = Server::pop();
				_solver.solveEvent(ev);
			} while (ev != NULL);
		}


	private:
		static EventSolver _solver;
	};

	template<typename MyEvent>
	void Associate(Action *action) {
		Solver::Associate(MyEvent::getId(), action);
	}
}


class SolveMouseMotionEvent : public Action {
public:
	SolveMouseMotionEvent(Window **w, std::list<Widget*> *lst) : _main(w), _new(lst) {}
	virtual ~SolveMouseMotionEvent() {
		for(auto w : _old)
			dynamic_cast<CallbackWidget *>(w)->onFocusLooseCallback();
	}

	virtual void triggerAction(Event *ev) {
		MouseMotionEvent *mouse = dynamic_cast<MouseMotionEvent *>(ev);
		_new->clear();
		((*_main)->getWidgetAt(mouse->getPos(), *_new));
		for (auto w : *_new) 
			dynamic_cast<CallbackWidget *>(w)->onFocusCallback(&_old);
		for (auto w : _old) {
			CallbackWidget *cw = dynamic_cast<CallbackWidget *>(w);
			if (!cw->getFocus())
				cw->onFocusLooseCallback();
		}
		for (auto w : *_new){
			dynamic_cast<CallbackWidget *>(w)->unsetFocus();
		}
		_old = *_new;
	}

private:
	Window** _main;
	std::list<Widget*> *_new, _old;
};


class SolveMouseClickEvent : public Action {
public:
	SolveMouseClickEvent(std::list<Widget*> *lst) : _new(lst) {}
	virtual ~SolveMouseClickEvent() {}
	
	virtual void triggerAction(Event *ev) {
		MouseClickEvent *mouse = dynamic_cast<MouseClickEvent *>(ev);
		if (!mouse->getDown()) {
			for (auto w : *_new){
				CallbackWidget *cw = dynamic_cast<CallbackWidget *>(w);
				if (cw->loopbackAction()) {
					Log(LWARNING, "Button ", (int)mouse->getValue(), "clicked");
					cw->callback((void*)mouse->getValue());
				}
				else {
					mouse->setWidget(w);
					mouse->forward(true);
					PublicSdlEventServer::push(mouse);
				}
			}
		}
	}

private:
	std::list<Widget*> *_new;
};

class SolveKeyPressedEvent : public Action {
public:
	SolveKeyPressedEvent(TextFieldWidget **w) : _w(w) {}


	virtual void triggerAction(Event *ev) {
		if (*_w == NULL){
			ev->forward(true);
			PublicSdlEventServer::push(ev);
		}
		else {
			KeyPressedEvent *key = dynamic_cast<KeyPressedEvent *>(ev);
			if(key->getDown()){
				if (key->getValue() == SDLK_ESCAPE || key->getValue() == SDLK_RETURN) {
					(*_w)->setColor(); 
					(*_w)->doForward();
					*_w = NULL;
					Log(LINFO, "Edition finished");
				}
				else if (key->getValue() == SDLK_BACKSPACE) {
					(*_w)->popLetter();
				}
				else {
					char c;
					if (key->getValue() == 59)
						c = '.';
					else c = ('a' + key->getValue() - 97);
					Log(LDEBUG, "Key ", key->getValue(), " ", (int)c);
					(*_w)->addLetter(c);
				}
			}
		}
	}

private:
	TextFieldWidget **_w;
};


class SolveEditTextFieldEvent : public Action {
public:
	SolveEditTextFieldEvent(TextFieldWidget **w) : _w(w) {}

	virtual void triggerAction(Event *ev) {
		EditTextFieldEvent *edit = dynamic_cast<EditTextFieldEvent *>(ev);
		(*_w) = static_cast<TextFieldWidget*>(edit->getWidget());
		(*_w)->setColor(150, 150, 200);
	}



private:
	TextFieldWidget **_w;
};

class SolveCallbackWidgetRemovedEvent : public Action {
public:
	SolveCallbackWidgetRemovedEvent(std::list<Widget*> *lst) : _new(lst) {}
	virtual ~SolveCallbackWidgetRemovedEvent() {}

	virtual void triggerAction(Event *ev) {
		CallbackWidgetRemovedEvent *cbrem = dynamic_cast<CallbackWidgetRemovedEvent *>(ev);
		_new->remove((Widget*)cbrem->getWidget());
	}

private:
	std::list<Widget*> *_new;
};
