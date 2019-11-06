#include "LetterLib.h"

SDL_Renderer *LetterLib::_renderer = NULL;
std::unordered_map<std::string, LetterLib*> LetterLib::_lib = std::unordered_map<std::string, LetterLib*>();

LetterLib::LetterLib() : 
Compliant("letterLib"),
_biggest({0, 0, 0, 0})
{
}


LetterLib::~LetterLib()
{
}


void LetterLib::split(const std::string & s, std::vector<std::string> &sentence, std::vector<ge_d > &size)
{
	std::string str(" ");
	sentence.clear();
	size.clear();
	ge_d lengh = _letter[' ']->getWidth();
	Letter *l;
	for (auto c : s) {
		if (c == ' ') {
			sentence.push_back(str);
			str = " ";
			size.push_back(lengh);
			lengh = _letter[' ']->getWidth();
		}
		else {
			l = getLetter(c);
			if (l != NULL) {
				lengh += l->getWidth();
				str += c;
			}
		}
	}
	if(str!=" "){
		sentence.push_back(str);
		size.push_back(lengh);
	}
}

std::string LetterLib::XMLName() const
{
	return staticXMLName();
}

std::string LetterLib::staticXMLName()
{
	return "letterLib";
}

std::string LetterLib::getLibName() const
{
	return _name;
}


void LetterLib::loadLib(const std::string & file)
{
	SDL_Surface *surface = TextureDataBase::requestSurface(file);
	SDL_SetColorKey(surface, 32, 0xffffffff);
	_biggest = { 0, 0, 0, 0 };
	ge_i align(-1), hmin(0), hmax(-1);

	if (surface != NULL) {
		while (!Surface::lineContainDark(surface, 0, surface->w, hmin) && hmin < surface->h) {
			hmin++;
		}
		hmax = hmin;
		while (Surface::lineContainDark(surface, 0, surface->w, hmax) && hmax < surface->h) {
			hmax++;
		}
		auto c = _symbolList.begin();
		SDL_Rect cut;
		cut.x = 0;
		cut.w = 0;
		SDL_Surface *dest = NULL;
		while (cut.x + cut.w < surface->w && c != _symbolList.end()) {
			if (Surface::columnContainDark(surface, cut.x + cut.w)) {
				cut.w++;
			}
			else if (cut.w > 0) {
				cut.y = hmin;
				cut.h = hmax;

				dest = SDL_CreateRGBSurface(0, cut.w * (1.0 + 2.0*_letterborder), cut.h  * (1.0 + 2.0* _lineborder), 32, R_MASK, G_MASK, B_MASK, A_MASK);
				SDL_Rect destrect = { (ge_i)(cut.w * _letterborder),  
					(ge_i)(cut.h*_lineborder), cut.w, cut.h };
				SDL_BlitSurface(surface, &cut, dest, &destrect);
				SDL_Texture *texture = SDL_CreateTextureFromSurface(_renderer, dest);
				SDL_FreeSurface(dest);
				createLetter(*c, texture, destrect);
				_biggest.w = max(_biggest.w, destrect.w);
				_biggest.w = max(_biggest.h, destrect.h);
				cut.x += cut.w;
				cut.w = 0;
				++c;
			}
			else {
				cut.x++;
			}
		}
		SDL_Rect destrect = { 0, 0, _spaceSize * _biggest.w , _biggest.h };
		dest = SDL_CreateRGBSurface(0, 1, 1, 32, R_MASK, G_MASK, B_MASK, A_MASK);
		SDL_Texture *texture = SDL_CreateTextureFromSurface(_renderer, dest);
		SDL_FreeSurface(dest);

		createLetter(' ', texture, destrect);

		for (auto l : _letter) {
			l.second->computeRelative(_biggest);
		}

		for (auto c : _convertSpace)
			setAsSpace(c);
	}
}

void LetterLib::associate()
{
	XMLAssociateField("symboles", new XML::String(&_symbolList));
	XMLAssociateField("file", new XML::String(&_file));
	XMLAssociateField("lineBorder", new XML::Double(&_lineborder));
	XMLAssociateField("letterBorder", new XML::Double(&_letterborder));
	XMLAssociateField("spaceSize", new XML::Double(&_spaceSize));
	XMLAssociateField("convertSpace", new XML::String(&_convertSpace));
}

void LetterLib::postLoading()
{
	loadLib(_file);
}

const ge_d & LetterLib::getSpaceSize() const
{
	return _spaceSize;
}

void LetterLib::setAsSpace(const char & c)
{
	_letter.insert(std::make_pair(c, _letter[' ']));
}

bool LetterLib::letterExist(const char & c)
{
	if (getLetter(c) == NULL)
		return false;
	return true;
}

LetterLib::Letter * LetterLib::getLetter(const char & c)
{
	auto it = _letter.find(c);
	if (it == _letter.end())
		return NULL;
	return it->second;
}

void LetterLib::setRenderer(SDL_Renderer * renderer)
{
	_renderer = renderer;
}


LetterLib * LetterLib::getLib(const std::string & name)
{
	auto it = _lib.find(name);
	if (it == _lib.end()) {
		LetterLib *lib = new LetterLib();
		lib->load(name);
		lib->_name = name;
		if (lib->_letter.size() == 0) {
			delete lib;
			lib = NULL;
		}
		_lib.insert(std::make_pair(name, lib));
		return lib;
	}
	else
		return it->second;
}

LetterLib * LetterLib::getDefaultLib()
{
	if (_lib.begin() != _lib.end())
		return _lib.begin()->second;
	return NULL;
}

void LetterLib::createLetter(const char & c, SDL_Texture * texture, const SDL_Rect & rect)
{
	std::string id = _file + "::" + c;
	Letter *l = new LetterLib::Letter(texture, rect,id);
	_letter.insert(std::make_pair(c, l));
	AnimationDataBase::addAnimation(id, l);
}

LetterLib::Letter::Letter(SDL_Texture * texture, const SDL_Rect & dim, const std::string &id) : Animation(), _dim(dim), _id(id)
{
	addTexture(texture);
}

void LetterLib::Letter::computeRelative(const SDL_Rect & biggest)
{
	_prop = square_d(0.0, 0.0, 0.0, 0.0);
	_prop.dim.w = (ge_d)_dim.w / biggest.w;
	_prop.dim.h = (ge_d)_dim.h / biggest.h;
}

square_d LetterLib::Letter::getRelative() const
{
	return _prop;
}

const ge_d & LetterLib::Letter::getWidth() const
{
	return _prop.dim.w;
}

const ge_d & LetterLib::Letter::getHeigh() const
{
	return _prop.dim.h;
}

std::string LetterLib::Letter::getFilename(void *v)
{
	return _id;
}
