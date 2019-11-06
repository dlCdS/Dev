#pragma once
#include "CallbackWidget.h"

class GetColorWidget :
	public CallbackWidget
{
public:
	GetColorWidget();
	virtual ~GetColorWidget();

	virtual void draw(SDL_Renderer *renderer);
	void setSize(const ge_pi &size);
	void setTexture(SDL_Texture *texture);

	std::string XMLName() const;
	static std::string staticXMLName();
	Color getColor() const;

protected:
	virtual void postComputeRelative(const ge_pi &pen); 
	void onFocus(void*v);
	void onClick(void*v);
	virtual void noCallback(void *v);

	ge_pi _size;
	SDL_Texture *_texture;
	SDL_Surface *_surface;
};

