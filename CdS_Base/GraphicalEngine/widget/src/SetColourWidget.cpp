#include "SetColourWidget.h"



SetColourWidget::SetColourWidget() :
	CallbackWidget(VoidedCallbackFunction(SetColourWidget, onClick), true,
		VoidedCallbackFunction(SetColourWidget, onFocus), true),
	Widget(NULL, false, true, true, true, false, square_d(-1, -1, -1, -1)),
	_surface(NULL),
	_texture(NULL)
{
}


SetColourWidget::~SetColourWidget()
{
}

void SetColourWidget::draw(SDL_Renderer * renderer)
{
	_color.setColor(renderer);
	SDL_RenderFillRect(renderer, &_seen);
	SDL_RenderCopy(renderer, _texture, NULL, &_seen);
}

void SetColourWidget::setSize(const ge_pi & size)
{
	_size = size;
}

void SetColourWidget::generateSurface()
{
	SDL_FreeSurface(_surface);
	_surface = Surface::getSurface(_size.w, _size.h);
	updateTexture();
	sqdHasChanged();
}

std::string SetColourWidget::XMLName() const
{
	return staticXMLName();
}

std::string SetColourWidget::staticXMLName()
{
	return "SetColourWidget";
}

Color SetColourWidget::getColor() const
{
	return _color;
}

void SetColourWidget::setPixelColor(const ge_i & x, const ge_i & y, const Color & color)
{
	Surface::putpixel(_surface, x, y, color.get());
}

void SetColourWidget::setRectColor(const SDL_Rect& rect, const Color& color)
{
	color.setColor(Renderer);

	SDL_SetTextureBlendMode(_texture, SDL_BLENDMODE_BLEND);
	SDL_SetRenderTarget(Renderer, _texture);
	SDL_RenderFillRect(Renderer, &rect);
	SDL_SetRenderTarget(Renderer, NULL);
}

void SetColourWidget::updateTexture()
{
	SDL_DestroyTexture(_texture);
	_texture = TextureDataBase::getTexture(_surface, false);
}

void SetColourWidget::setupForRect()
{
	SDL_DestroyTexture(_texture);
	_texture = SDL_CreateTexture(Renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_TARGET, _size.w, _size.h);
}

SDL_Surface * SetColourWidget::getSurface()
{
	return _surface;
}

void SetColourWidget::postComputeRelative(const ge_pi & pen)
{
	if (_sqd.dim.w < 0.0)
		_rel.w = _size.w;
	if (_sqd.dim.h < 0.0)
		_rel.h = _size.h;
}


void SetColourWidget::onFocus(void * v)
{
	if (_surface != NULL) {
		_pos = { _surface->w*(_pos.x - _abs.x) / _size.w, _surface->h*(_pos.y - _abs.y) / _size.h };
		_color = Surface::getpixel(_surface, _pos.x, _pos.y);
	}
}

void SetColourWidget::onClick(void * v)
{
}

void SetColourWidget::noCallback(void * v)
{
}

