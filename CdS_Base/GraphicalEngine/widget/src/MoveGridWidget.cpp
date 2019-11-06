#include "MoveGridWidget.h"



MoveGridWidget::MoveGridWidget(const ge_pi &gridDim) :
	CallbackWidget(VoidedCallbackFunction(MoveGridWidget, noCallback), true,
		VoidedCallbackFunction(MoveGridWidget, onFocus), true,
		VoidedCallbackFunction(MoveGridWidget, onLooseFocus), true),
	GridWidget(gridDim),
	Widget(NULL, 0, 0, 1, 0, 0, square_d(-1, -1, -1, -1)),
	_moveArea(20),
	_doMove(false)
{
}


MoveGridWidget::~MoveGridWidget()
{
}

void MoveGridWidget::addWidget(Widget * widget)
{
	GridWidget::addWidget(widget);
}

bool MoveGridWidget::addWidgetAt(Widget * widget, const ge_pi & pos)
{
	_child.push_back(widget);
	widget->setParent(this);
	if (!_noChild && _cur.w < _gridDim.w && _cur.h < _gridDim.h) {
		_layers[_layers.size() - 1]->add(widget);
		widget->setRelativeProportion(square_d(
			1.0 * _cur.w / _gridDim.w,
			1.0 * _cur.h / _gridDim.h,
			1.0 / _gridDim.w,
			1.0 / _gridDim.h));
		if (++_cur.w >= _gridDim.w)
			_cur = { 0, _cur.h + 1 };
		return true;
	}
	else return false;
}


std::string MoveGridWidget::XMLName() const
{
	return staticXMLName();
}

std::string MoveGridWidget::staticXMLName()
{
	return "MoveGridWidget";
}

void MoveGridWidget::draw(SDL_Renderer * renderer)
{
	Widget::draw(renderer);
	if (_doMove)
		move();
}

bool MoveGridWidget::getWidgetAt(const SDL_Point & pos, std::list<Widget*>& list)
{
	return CallbackWidget::getWidgetAt(pos, list);
}



void MoveGridWidget::move()
{
	if (_move.w < 0 && _abstract.x + _abstract.w  <= _abs.w)
		_move.w = 0;
	else if (_move.w > 0 && _abstract.x >= 0)
		_move.w = 0;

	if (_move.h < 0 && _abstract.y + _abstract.h <= _abs.h)
		_move.h = 0;
	else if (_move.h > 0 && _abstract.y >= 0)
		_move.h = 0;
	if (_move.w == 0 && _move.h == 0)
		_doMove = false;
	if (_doMove) {
		_abstract.x += _move.w;
		_abstract.y += _move.h;
		relHasChanged();
	}
}

void MoveGridWidget::onFocus(void * v)
{
	_move = { 0, 0 };
	if (_pos.x - _abs.x <= _moveArea)
		_move.w = _moveArea + _abs.x - _pos.x;
	else if (_abs.x + _abs.w - _pos.x <= _moveArea)
		_move.w = _abs.x + _abs.w - _pos.x - _moveArea;

	if (_pos.y - _abs.y <= _moveArea)
		_move.h = _moveArea + _abs.y - _pos.y;
	else if (_abs.y + _abs.h - _pos.y <= _moveArea)
		_move.h = _abs.y + _abs.h - _pos.y - _moveArea;
	_doMove = true;
}

void MoveGridWidget::onLooseFocus(void * v)
{
	_doMove = false;
}

void MoveGridWidget::noCallback(void * v)
{
}

void MoveGridWidget::associate()
{
	GridWidget::associate();
}
