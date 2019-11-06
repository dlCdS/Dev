#include "Surface.h"
#include "Defines.h"
#include <Clocks.h>

Uint32 Surface::getpixel(SDL_Surface * surface, int x, int y)
{
	int bpp = surface->format->BytesPerPixel;
	/* Here p is the address to the pixel we want to retrieve */
	Uint8 *p = (Uint8 *)surface->pixels + y * surface->pitch + x * bpp;

	switch (bpp) {
	case 1:
		return *p;
		break;

	case 2:
		return *(Uint16 *)p;
		break;

	case 3:
		if (SDL_BYTEORDER == SDL_BIG_ENDIAN)
			return p[0] << 16 | p[1] << 8 | p[2];
		else
			return p[0] | p[1] << 8 | p[2] << 16;
		break;

	case 4:
		return *(Uint32 *)p;
		break;

	default:
		return 0;       /* shouldn't happen, but avoids warnings */
	}
}

void Surface::putpixel(SDL_Surface * surface, int x, int y, Uint32 pixel)
{
	int bpp = surface->format->BytesPerPixel;
	/* Here p is the address to the pixel we want to set */
	Uint8 *p = (Uint8 *)surface->pixels + y * surface->pitch + x * bpp;

	switch (bpp) {
	case 1:
		*p = pixel;
		break;

	case 2:
		*(Uint16 *)p = pixel;
		break;

	case 3:
		if (SDL_BYTEORDER == SDL_BIG_ENDIAN) {
			p[0] = (pixel >> 16) & 0xff;
			p[1] = (pixel >> 8) & 0xff;
			p[2] = pixel & 0xff;
		}
		else {
			p[0] = pixel & 0xff;
			p[1] = (pixel >> 8) & 0xff;
			p[2] = (pixel >> 16) & 0xff;
		}
		break;

	case 4:
		*(Uint32 *)p = pixel;
		break;
	}
}

SDL_Surface * Surface::getSurface(SDL_Renderer * renderer, SDL_Texture * texture)
{
	SDL_Texture *ren_tex;
	SDL_Texture* target = SDL_GetRenderTarget(renderer);
	SDL_SetRenderTarget(renderer, texture);
	SDL_RenderClear(renderer);
	int width, height;
	SDL_QueryTexture(texture, NULL, NULL, &width, &height);
	ren_tex = SDL_CreateTexture(renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_TARGET, width, height);
	SDL_SetRenderTarget(renderer, ren_tex);

	SDL_SetRenderDrawColor(renderer, 0x00, 0x00, 0x00, 0x00);

	SDL_Surface* surface = getSurface(width, height);
	SDL_RenderCopy(renderer, texture, NULL, NULL);
	SDL_RenderReadPixels(renderer, NULL, surface->format->format, surface->pixels, surface->pitch);
	SDL_SetRenderTarget(renderer, target);
	return surface;
}

SDL_Surface * Surface::getSurface(const ge_i & width, const ge_i & heigh)
{
	return SDL_CreateRGBSurface(0, width, heigh, 32, R_MASK, G_MASK, B_MASK, A_MASK);
}

SDL_Surface * Surface::getColorTriangle()
{
	const int surfSize = 510, size = 3, offset = 255;
	SDL_Surface *surface = getSurface(surfSize, surfSize);
	int s[size];
	const int v1[] = { 0.5, 1, -1 }, v2[] = { 1, -1, -0.5 };
	int x, y;
	Uint32 c;
	s[0] = 255;
	for(s[1] =0; s[1] <256 && true; s[1]++)
		for (s[2] = 0; s[2] < 256; s[2]++) {
			x = offset + Math::ScalarProd<int>(v1, s, size);
			y = offset + Math::ScalarProd<int>(v2, s, size) + x / 2 - surfSize / 4;
			putpixel(surface, x, y, Color(s[0], s[1], s[2]).get());
		}
	s[1] = 255;
	for (s[0] = 0; s[0] < 255 && true; s[0]++)
		for (s[2] = 0; s[2] < 256; s[2]++) {
			x = offset + Math::ScalarProd<int>(v1, s, size);
			y = offset + Math::ScalarProd<int>(v2, s, size) + x / 2 - surfSize / 4;
			putpixel(surface, x, y, Color(s[0], s[1], s[2]).get());
		}
	s[2] = 255;
	for (s[1] = 0; s[1] < 255 && true; s[1]++)
		for (s[0] = 0; s[0] < 255; s[0]++) {
			x = offset + Math::ScalarProd<int>(v1, s, size);
			y = offset + Math::ScalarProd<int>(v2, s, size) + x/2 - surfSize / 4;
			putpixel(surface, x, y, Color(s[0], s[1], s[2]).get());
		}

	return surface;
}

SDL_Surface * Surface::getGradient(const ColorS &color)
{
	SDL_Surface *surface = getSurface(2, 256);
	ColorS c = color;
	for (int i = 0; i < 256; i++) {
		c._s = (1.0 - (ge_d)i / 256);
		for(int j=0; j<2; j++ )
			putpixel(surface, j, i, c.getColor().get());
	}
	return surface;
}

SDL_Surface * Surface::getDefault()
{
	SDL_Surface *surface = getSurface(DEFAULT_SURFACE_SIZE, DEFAULT_SURFACE_SIZE);
	return surface;
}

bool Surface::columnContainDark(SDL_Surface * surface, int x)
{
	for (int y = 0; y < surface->h; y++) {
		Uint32 pix = getpixel(surface, x, y);
		pix = getColorNorm(pix);
		if (pix < 10)
			return true;
	}
	return false;
}

bool Surface::lineContainDark(SDL_Surface * surface, int from, int to, int y)
{
	bool found(false);
	for (int x = from; x < to; x++) {
		Uint32 pix = getpixel(surface, x, y);
		if (getColorNorm(pix) < 10)
			found = true;
		else
			putpixel(surface, x, y, 0xffffffff);
	}
	return found;
}

Uint32 Surface::getColorNorm(const Uint32 & color)
{
	return (color & B_MASK);
}
