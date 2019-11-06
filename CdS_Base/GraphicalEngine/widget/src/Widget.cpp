#include "Widget.h"


SDL_Renderer* Widget::Renderer = NULL;

Widget::Widget() :
	Parsable(),
	_scrollable(false),
	_autoFeedLine(false),
	_clickable(false),
	_noChild(false),
	_parent(NULL),
	_xmlsqd(&_sqd),
	_sqdHasChanged(true),
	_relHasChanged(true),
	_shared(false),
	_abstract({0, 0, 0, 0})
{
}

Widget::Widget(Widget * parent) :
	Parsable(),
	_parent(parent),
	_xmlsqd(&_sqd),
	_sqdHasChanged(true),
	_relHasChanged(true),
	_shared(false),
	_abstract({ 0, 0, 0, 0 })
{
}

Widget::Widget(Widget * parent, const bool scrollable, const bool autoFeedLine, const bool clickable, const bool noChild, const bool &onePerLine) :
	Parsable(),
	_scrollable(scrollable),
	_autoFeedLine(autoFeedLine),
	_clickable(clickable),
	_noChild(noChild),
	_onePerLine(onePerLine),
	_sqd(square_d(-1, -1, -1, -1)),
	_xmlsqd(&_sqd),
	_sqdHasChanged(true),
	_relHasChanged(true),
	_shared(false),
	_abstract({ 0, 0, 0, 0 })
{
}

Widget::Widget(Widget * parent, const bool scrollable, const bool autoFeedLine, 
	const bool clickable, const bool noChild, const bool &onePerLine, const square_d & dim) :
	Parsable(),
	_scrollable(scrollable),
	_autoFeedLine(autoFeedLine),
	_clickable(clickable),
	_noChild(noChild),
	_onePerLine(onePerLine),
	_sqd(dim),
	_xmlsqd(&_sqd),
	_sqdHasChanged(true),
	_relHasChanged(true),
	_shared(false),
	_abstract({ 0, 0, 0, 0 })
{
}

Widget::~Widget()
{
	if (_parent != NULL)
		_parent->removeChild(this);
	freeAll();
}

void Widget::computeRelative(const ge_pi & pen)
{
	SDL_Rect oldrel = _rel;
	_rel = { 0, 0, 0, 0 };

	commonComputeRelative(pen);

	postComputeRelative(pen);
	
	_relHasChanged = true;
}

void Widget::relHasChanged(const bool &propagateup, const bool &propagatedown)
{
	_relHasChanged = true;
	if(propagatedown)
		for (auto c : _child)
			c->relHasChanged(false, false);
	if (_parent != NULL && propagateup)
		_parent->relHasChanged();
}

void Widget::sqdHasChanged(const bool &propagateup, const bool &propagatedown)
{
	_sqdHasChanged = true;
	if (propagatedown)
		for (auto c : _child)
			c->sqdHasChanged(false, false);
	if (_parent != NULL && propagateup)
		_parent->sqdHasChanged();
}

void Widget::setRenderer(SDL_Renderer* renderer)
{
	Renderer = renderer;
}

XML::Parsable * Widget::XMLAddWidgetLayer(void * v)
{
	addLayer();
	return _layers[_layers.size() - 1];
}

void Widget::associate()
{
	XMLAssociateField("autoFeedLine", new XML::Bool(&_autoFeedLine));
	XMLAssociateField("clickable", new XML::Bool(&_clickable));
	XMLAssociateField("noChild", new XML::Bool(&_noChild));

	XMLAssociateSubBeacon("proportion", &_xmlsqd);
	XMLAssociateSubBeacon("layers", _layers.begin(), _layers.end(), SubBeaconLoadFunction(Widget, XMLAddWidgetLayer));

}

void Widget::computeAbsolute(const SDL_Rect &container, const SDL_Rect &seen, const SDL_Rect &offset)
{
	if (_relHasChanged) {
		_abs = {
			_rel.x + container.x + offset.x,
			_rel.y + container.y + offset.y,
			_rel.w,
			_rel.h
		};
		SDL_IntersectRect(&seen, &_abs, &_seen);
		if(_seen.w > 0 && _seen.h>0){
			for (auto l : _layers)
				for (auto w : l->_wid){
					w->relHasChanged(false);
					w->computeAbsolute(_abs, _seen, _abstract);
				}
		}
		
		_relHasChanged = false;
	}
}

void Widget::drawWidgets(SDL_Renderer *renderer)
{
	if (_sqdHasChanged)
		computeChild();
	if (_relHasChanged)
		computeAbsolute(_parent->_abs, _parent->_abs, _abstract);
	if(_seen.h > 0 && _seen.w > 0) {
		for (auto l : _layers)
			for (auto w : l->_wid)
				w->drawWidgets(renderer);
		draw(renderer);
	}
}

const bool & Widget::isShared() const
{
	return _shared;
}

void Widget::removeChild(Widget * w)
{
	auto it = std::find(_child.begin(), _child.end(), w);
	if (it != _child.end()) {
		*it = _child[_child.size() - 1];
		_child.pop_back();
		bool found(false);
		for (auto l = _layers.begin(); l != _layers.end() && !found; ++l) {
			auto mw = std::find((*l)->_wid.begin(), (*l)->_wid.end(), w);
			if (mw != (*l)->_wid.end()) {
				(*l)->_wid.erase(mw);
				found = true;
			}
		}
	}
}

void Widget::draw(SDL_Renderer *renderer)
{
	_color.setColor(renderer);
	SDL_RenderDrawRect(renderer, &_seen);
}

void Widget::freeAll()
{
	for (auto w : _child) {
		if(!w->_shared)
			delete w;
		else w->setParent(NULL);
	}
	_child.clear();
	_layers.clear();
}

void Widget::addWidget(Widget * widget)
{
	if (_layers.size() == 0)
		addLayer();
	_child.push_back(widget);
	if (!_noChild) {
		_layers[_layers.size() - 1]->add(widget);
		widget->_parent = this;
	}
}

void Widget::setParent(Widget * parent)
{
	_parent = parent;
}

void Widget::addLayer()
{
	if(!_noChild) {
		_layers.push_back(new WidgetLayer(this));
	}
}

void Widget::setRelativeProportion(const square_d & sqd)
{
	_sqd = sqd;
}

void Widget::setRelativeSquare(const SDL_Rect & rel)
{
	_rel = rel;
}

void Widget::setAbstractSquare(const SDL_Rect & abs)
{
	_abstract = abs;
}

void Widget::setAbstractPos(const ge_i & x, const ge_i & y)
{
	_abstract.x = x;
	_abstract.y = y;
}




void Widget::commonComputeRelative(const ge_pi &pen)
{
	SDL_Rect pr = { 0, 0, 0, 0 };
	getFirstNotEmptyRelRect(pr);
	if (_sqd.pos.w < 0) 
		_rel.x = pen.w;
	else _rel.x = _sqd.pos.w*pr.w;

	if (_sqd.pos.h < 0) 
		_rel.y = pen.h;
	else _rel.y = _sqd.pos.h*pr.h;

	if (_sqd.dim.w >= 0) 
		_rel.w = _sqd.dim.w*pr.w;
	else _rel.w = pr.w;

	if (_sqd.dim.h >= 0) 
		_rel.h = _sqd.dim.h*pr.h;
	else _rel.h = pr.h;

	computeChild();

	if (_sqd.dim.w < 0)
		_rel.w = _rdcorner.w;

	if (_sqd.dim.h < 0)
		_rel.h = _rdcorner.h;

}

void Widget::postComputeRelative(const ge_pi & pen)
{
}

void Widget::computeChild()
{	
	_pen = { 0, 0 };
	_rdcorner = { 0, 0 };
	SDL_Rect memrel = _rel;
	if (_abstract.h != 0 && _abstract.w != 0)
		_rel = _abstract;

	for (auto c : _child) {
		c->computeRelative(_pen);
		SDL_Rect r = c->getRelativeRect();
		if (_onePerLine) {
			_pen.w = 0;
			_pen.h = r.y + r.h;
		}
		else if (r.x + r.w > _rel.w) {
			_pen.w = 0;
			_pen.h = r.y + r.h;
			if (_autoFeedLine) {
				c->computeRelative(_pen);
				r = c->getRelativeRect();
				_pen.w = r.x + r.w;
			}
		}
		else if (r.x + r.w == _rel.w) {
			_pen.w = 0;
			_pen.h = r.y + r.h;
		}
		else {
			_pen.w = r.x + r.w;
		}
		_rdcorner.w = max(_rdcorner.w, r.w + r.x);
		_rdcorner.h = max(_rdcorner.h, r.h + r.y);
	}
	_sqdHasChanged = false;
	_rel = memrel;
}

bool Widget::getWidgetAt(const SDL_Point & pos, std::list<Widget*> &list)
{
 	if (SDL_PointInRect(&pos, &_abs)) {
		if (_clickable)
			list.push_back(this);
		for (auto l = _layers.rbegin(); l != _layers.rend(); ++l)
			for (auto w = (*l)->_wid.begin(); w != (*l)->_wid.end(); ++w)
				if((*w)->getWidgetAt(pos, list))
					return true;
		return true;
	} 
	return false;
}

SDL_Rect Widget::getAbsRect() const
{
	return _abs;
}

SDL_Rect Widget::getRelativeRect() const
{
	return _rel;
}

void Widget::getFirstNotEmptyRelRect(SDL_Rect & rect)
{
	if (rect.w <= 0)
 		rect.w = _rel.w;
	if (rect.h <= 0)
		rect.h = _rel.h;
	if (rect.w <= 0 || rect.h <= 0)
		_parent->getFirstNotEmptyRelRect(rect);
}

bool Widget::RectEqual(const SDL_Rect & r1, const SDL_Rect & r2)
{
	if (r1.w == r2.w &&
		r1.h == r2.h &&
		r1.x == r2.x &&
		r1.y == r2.y)
		return true;
	return false;
}

void Widget::updateContent()
{
}

void Widget::setColor(const Uint8 & r, const Uint8 & g, const Uint8 & b)
{
	_color = Color(r, g, b);
}

Color * Widget::getColorInstance()
{
	return &_color;
}

void Widget::setShared()
{
	_shared = true;
}


ge_pi Widget::getPen() const
{
	return _pen;
}

void Color::setColor(SDL_Renderer * renderer, const Uint8 & alpha) const
{
	SDL_SetRenderDrawColor(renderer, _r, _g, _b, alpha);
}
