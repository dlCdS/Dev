#pragma once
#include <SDL.h>
#include "Defines.h"

#define DEFAULT_SURFACE_SIZE 135
#define BYTE_SIZE 256
#define COLORED_TRIANGLE "ColoredTriangle"

namespace Surface
{

	Uint32 getpixel(SDL_Surface *surface, int x, int y);
	void putpixel(SDL_Surface *surface, int x, int y, Uint32 pixel);
	SDL_Surface *getSurface(SDL_Renderer* renderer, SDL_Texture* texture);
	SDL_Surface *getSurface(const ge_i &width, const ge_i &heigh);
	SDL_Surface *getColorTriangle();
	SDL_Surface *getGradient(const ColorS &color);
	SDL_Surface *getDefault();


	bool columnContainDark(SDL_Surface *surface, int x);
	bool lineContainDark(SDL_Surface *surface, int from, int to, int y);
	Uint32 getColorNorm(const Uint32 &color);

};


