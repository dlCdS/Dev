#include "TextureDataBase.h"


TextureDataBase TextureDataBase::_singleton = TextureDataBase();


TextureDataBase::TextureDataBase()
{
}


TextureDataBase::~TextureDataBase()
{
	freeAll();
}

SDL_Texture * TextureDataBase::getEmptyTexture(const Uint32 &color)
{
	SDL_Texture *texture = NULL;
	std::string name = DEFAULT_TEXTURE_NAME;
	name += std::to_string(color);
	texture = requestTexture(name);
	if (texture != NULL)
		return texture;
	texture = SDL_CreateTexture(_singleton._renderer,
	SDL_PIXELFORMAT_RGBA8888,
	SDL_TEXTUREACCESS_TARGET,
	90, 90);
	//SDL_Surface *surface = SDL_CreateRGBSurface(0, 90, 90, 8, R_MASK, G_MASK, B_MASK, A_MASK);

	//SDL_FillRect(surface, NULL, color);
	//texture = SDL_CreateTextureFromSurface(_renderer, surface);
	SDL_SetRenderTarget(_singleton._renderer, NULL);
	return texture;
}

SDL_Texture * TextureDataBase::requestTexture(const std::string &filename, const std::string &newname)
{
	if (_singleton._data.find(filename) != _singleton._data.end())
		return _singleton._data[filename];
	else if(newname != "" && _singleton._data.find(newname) != _singleton._data.end())
		return _singleton._data[newname];

	SDL_Surface *surface = requestSurface(filename);
	SDL_Texture *texture = SDL_CreateTextureFromSurface(_singleton._renderer, surface);
	SDL_FreeSurface(surface);
	if(newname=="")
		_singleton._data.insert(std::make_pair(filename, texture));
	else
		_singleton._data.insert(std::make_pair(newname, texture));
	return texture;
}

SDL_Surface * TextureDataBase::requestSurface(const std::string & filename)
{
	SDL_Surface *surface = NULL;
	

#ifdef USE_IMAGE_LOADER

	surface = IMG_Load(filename.c_str());
	if (surface == NULL)
	{
		printf("Unable to load image %s! SDL_image Error: %s\n", filename.c_str(), IMG_GetError());
	}


#else
	surface = SDL_LoadBMP(filename.c_str());

	if (!surface) {
		_data.insert(std::make_pair(filename, (SDL_Texture*)NULL));
		if (!surface) {
			std::cout << "TextureDataBase loading failed " << filename << std::endl;
		}
		return NULL;
	}
	SDL_SetColorKey(surface,
		-1,
		MASK);
	surface->format->Amask = 0xff000000;
	surface->format->Ashift = 24;

#endif
	return surface;
}

SDL_Texture * TextureDataBase::addTexture(const std::string & name, SDL_Surface * surface)
{
	SDL_Texture *texture = SDL_CreateTextureFromSurface(_singleton._renderer, surface);
	SDL_FreeSurface(surface);
	_singleton._data.insert(std::make_pair(name, texture));
	return texture;
}

SDL_Texture * TextureDataBase::getTexture(SDL_Surface * surface, const bool &deleteSurface)
{
	SDL_Texture *tex = SDL_CreateTextureFromSurface(_singleton._renderer, surface);
	if(deleteSurface)
		SDL_FreeSurface(surface);
	return tex;
}

SDL_Renderer * TextureDataBase::getRenderer()
{
	return _singleton._renderer;
}

bool TextureDataBase::textureExists(const std::string & filename)
{
	if (_singleton._data.find(filename) != _singleton._data.end())
		return true;
	return false;
}

void TextureDataBase::setRenderer(SDL_Renderer * renderer, const SDL_PixelFormat &format)
{
	_singleton._renderer = renderer;
	_singleton._format = format;
}

void TextureDataBase::freeAll()
{
	_singleton.getKey(NULL);
	for (auto t = _singleton._data.begin(); t != _singleton._data.end(); ++t)
		SDL_DestroyTexture(t->second);
	_singleton._data.clear();
}

void TextureDataBase::updateTexture(const std::string & filename)
{
	if (_singleton._data.find(filename) != _singleton._data.end()) {
		SDL_DestroyTexture(_singleton._data[filename]);
		_singleton._data[filename] = NULL;
		SDL_Surface *surface = SDL_LoadBMP(filename.c_str());

		if (surface) {
			SDL_SetColorKey(surface,
				-1,
				MASK);
			//surface->format->Amask = 0xFF000000;
			//surface->format->Ashift = 24;
			_singleton._data[filename] = SDL_CreateTextureFromSurface(_singleton._renderer, surface);
			SDL_FreeSurface(surface);
		}
		
	}
	else requestTexture(filename);
}

std::string TextureDataBase::getKey(const SDL_Texture * animation)
{
	auto name = std::find_if(
		_singleton._data.begin(),
		_singleton._data.end(),
		[&animation](const  std::unordered_map<std::string, SDL_Texture*>::value_type& vt)
	{ return vt.second == animation; });

	if (_singleton._data.end() != name)
		return name->first;
	else
		return "NULL";
}

void TextureDataBase::saveTexture(const std::string &file, SDL_Renderer * renderer, SDL_Texture * texture)
{
	SDL_Surface *surface = Surface::getSurface(renderer, texture);
#ifdef USE_IMAGE_LOADER
	IMG_SavePNG(surface, file.c_str());
#else
	IMG_SavePNG(surface, file.c_str());
#endif
	SDL_FreeSurface(surface);
}
