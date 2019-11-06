#pragma once
#include "CallbackWidget.h"
#include "GridWidget.h"

class MoveGridWidget :
	public CallbackWidget, public GridWidget
{
public:
	MoveGridWidget(const ge_pi &gridDim);
	~MoveGridWidget();


	virtual void addWidget(Widget *widget);
	virtual bool addWidgetAt(Widget *widget, const ge_pi &pos);

	virtual std::string XMLName() const;
	static std::string staticXMLName();

	virtual void draw(SDL_Renderer *renderer);
	virtual bool getWidgetAt(const SDL_Point & pos, std::list<Widget*> &list);

protected:
	void move();
	void onFocus(void*v);
	void onLooseFocus(void*v);
	void noCallback(void*v);
	virtual void associate();

	ge_i _moveArea;
	ge_pi _move;
	bool _doMove;
};

