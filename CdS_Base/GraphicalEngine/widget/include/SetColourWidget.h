#pragma once
#include "CallbackWidget.h"

class SetColourWidget :
	public CallbackWidget
{
public:
	SetColourWidget();
	~SetColourWidget();

	virtual void draw(SDL_Renderer *renderer);
	void setSize(const ge_pi &size);
	void generateSurface();

	std::string XMLName() const;
	static std::string staticXMLName();
	Color getColor() const;
	void setPixelColor(const ge_i &x, const ge_i &y, const Color &color);
	void setRectColor(const SDL_Rect &rect, const Color& color);
	void updateTexture();
	void setupForRect();
	SDL_Surface* getSurface();

protected:
	virtual void postComputeRelative(const ge_pi &pen);
	void onFocus(void*v);
	void onClick(void*v);
	virtual void noCallback(void *v);

	ge_pi _size;
	SDL_Texture *_texture;
	SDL_Surface *_surface;
};

