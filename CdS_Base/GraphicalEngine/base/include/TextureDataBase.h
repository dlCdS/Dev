#pragma once
#include <unordered_map>
#include <string>
#include <iostream>
#include "Surface.h"

#include <algorithm>
#include <SDL_image.h>

#define DEFAULT_TEXTURE_NAME "defaultTextureName";




#define USE_IMAGE_LOADER


class TextureDataBase
{
public:
	~TextureDataBase();

	static SDL_Texture* getEmptyTexture(const Uint32 &color);
	static SDL_Texture *requestTexture(const std::string &filename, const std::string &newname="");
	static SDL_Surface *requestSurface(const std::string &filename);
	static SDL_Texture *addTexture(const std::string &name, SDL_Surface *surface);
	static SDL_Texture *getTexture(SDL_Surface *surface, const bool &deleteSurface=true);
	static SDL_Renderer *getRenderer();
	static bool textureExists(const std::string &filename);
	static void setRenderer(SDL_Renderer *renderer, const SDL_PixelFormat &format);
	static void freeAll();
	static void updateTexture(const std::string &filename);

	static std::string getKey(const SDL_Texture *animation);
private:

	void saveTexture(const std::string &file, SDL_Renderer* renderer, SDL_Texture* texture);

	std::unordered_map<std::string, SDL_Texture*> _data;
	SDL_Renderer *_renderer;
	SDL_PixelFormat _format;

	static TextureDataBase _singleton;
	TextureDataBase();
};
