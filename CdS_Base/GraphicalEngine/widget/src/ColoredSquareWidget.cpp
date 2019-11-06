#include "ColoredSquareWidget.h"



ColoredSquareWidget::ColoredSquareWidget() : Widget(NULL, false, false, false, true, false)
{
}


ColoredSquareWidget::~ColoredSquareWidget()
{
}

void ColoredSquareWidget::setSize(const ge_pi & size)
{
	_size = size;
}

void ColoredSquareWidget::draw(SDL_Renderer * renderer)
{
	if(_seen.w > 0 && _seen.h >= 0){
		_color.setColor(renderer);
		SDL_RenderFillRect(renderer, &_seen);
	}
}

std::string ColoredSquareWidget::XMLName() const
{
	return staticXMLName();
}

std::string ColoredSquareWidget::staticXMLName()
{
	return "ColoredSquareWidget";
}

void ColoredSquareWidget::postComputeRelative(const ge_pi & pen)
{
	if (_sqd.dim.w < 0)
		_rel.w = _size.w;
	if (_sqd.dim.h < 0)
		_rel.h = _size.h;
}

void ColoredSquareWidget::associate()
{
	Widget::associate();
	XMLAssociateField("width", new XML::Integer(&_size.w));
	XMLAssociateField("heigh", new XML::Integer(&_size.h));
}
