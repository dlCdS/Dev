#include "GetColourWidget.h"



GetColorWidget::GetColorWidget() : 
	CallbackWidget(VoidedCallbackFunction(GetColorWidget, onClick), true,
		VoidedCallbackFunction(GetColorWidget, onFocus), true),
	Widget(NULL, false, true, true, true, false, square_d(-1, -1, -1, -1)),
	_surface(NULL),
	_texture(NULL)
{
}


GetColorWidget::~GetColorWidget()
{
}

void GetColorWidget::draw(SDL_Renderer * renderer)
{		
	_color.setColor(renderer);
	SDL_RenderFillRect(renderer, &_seen);
	SDL_RenderCopy(renderer, _texture, NULL, &_seen);
}

void GetColorWidget::setSize(const ge_pi & size)
{
	_size = size;
}

void GetColorWidget::setTexture(SDL_Texture * texture)
{
	_texture = texture;
	SDL_FreeSurface(_surface);
	_surface = NULL;
	_surface = Surface::getSurface(TextureDataBase::getRenderer(), _texture);
	sqdHasChanged();
}

std::string GetColorWidget::XMLName() const
{
	return staticXMLName();
}

std::string GetColorWidget::staticXMLName()
{
	return "GetColourWidget";
}

Color GetColorWidget::getColor() const
{
	return _color;
}

void GetColorWidget::postComputeRelative(const ge_pi & pen)
{
	if (_sqd.dim.w < 0.0)
		_rel.w = _size.w;
	if (_sqd.dim.h < 0.0)
		_rel.h = _size.h;
}


void GetColorWidget::onFocus(void * v)
{
	if (_surface != NULL) {
		_pos = { _surface->w*(_pos.x - _abs.x) / _size.w, _surface->h*(_pos.y - _abs.y) / _size.h };
		_color = Surface::getpixel(_surface, _pos.x, _pos.y);
	}
}

void GetColorWidget::onClick(void * v)
{
}

void GetColorWidget::noCallback(void * v)
{
}
