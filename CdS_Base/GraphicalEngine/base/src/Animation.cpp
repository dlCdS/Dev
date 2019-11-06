#include "Animation.h"

unsigned long Animation::_time = 0;

Animation::Animation()
{
}

Animation::Animation(SDL_Texture * texture)
{
	addTexture(texture);
}


Animation::~Animation()
{
}

Animation * Animation::getDefaultAnimation()
{
	Animation *animation = new Animation();
	for (int i = 0; i < 5; i++) {
		animation->_t.push_back(TextureDataBase::getEmptyTexture(0x00000000));
	}
	for (int i = 0; i < 5; i++) {
		animation->_t.push_back(TextureDataBase::getEmptyTexture(0xFFFFFFFF));
	}
	return animation;
}

void Animation::updateTime()
{
	_time++;
}

unsigned long &Animation::getTime()
{
	return _time;
}

SDL_Texture * Animation::getTexture(const unsigned long &time) const
{
	return _t[time%_t.size()];
}

void Animation::load(const char * filename, const std::string &newname)
{
	std::fstream f(filename, std::ios::in);
	if (f) {
		if (newname != "")
			_name = newname;
		else
			_name = filename;
		unsigned long duration, count(0);
		f >> duration;
		unsigned long t;
		std::string file;
		while (!f.eof()) {
			f >> t;
			while (count < t) {
				_t.push_back(TextureDataBase::requestTexture(file));
				count++;
			}
			f >> file;
		}
		while (count < duration) {
			_t.push_back(TextureDataBase::requestTexture(file));
			count++;
		}
	}
	f.close();
	
	std::cout << "animation loaded " << filename << std::endl;
}

void Animation::save(const char * filename)
{
	std::string name;
	std::fstream f(filename, std::ios::out | std::ios::trunc);
	if (f) {
		f << _t.size();
		for (unsigned long i = 0; i < _t.size(); i++) {
			if (i == 0) {
				name = TextureDataBase::getKey(_t[i]);
				if (name != "NULL") {
					f << std::endl << i;
					f << std::endl << name;
				}
			}
			else if (_t[i] != _t[i - 1]) {
				name = TextureDataBase::getKey(_t[i]);
				if (name != "NULL") {
					f << std::endl << i;
					f << std::endl << name;
				}
			}
		}
	}
	f.close();
}

unsigned long Animation::getDuration() const
{
	return _t.size();
}

void Animation::addFrame()
{
	_t.push_back(_t[_t.size() - 1]);
}

void Animation::removeFrame()
{
	if (_t.size() > 1)
		_t.pop_back();
}

void Animation::addTexture(const std::string &filename)
{
	_t.push_back(TextureDataBase::requestTexture(filename));
}

void Animation::addTexture(SDL_Texture * t)
{
	_t.push_back(t);
}

bool Animation::textureFirstOccurence(const unsigned long &time) const
{
	if (getTexture(time) == getTexture(time - 1))
		return false;
	return true;
}

int Animation::getNumberOfTexture() const
{
	std::unordered_set<SDL_Texture*> set;
	for (int i = 0; i < _t.size(); i++)
		set.insert(_t[i]);
	return set.size();
}

std::string Animation::getFilename()
{
	return _name;
}

std::string Animation::XMLName() const
{
	return staticXMLName();
}

std::string Animation::staticXMLName()
{
	return "animation";
}


void Animation::associate()
{
	XMLAssociateField("name", new XML::String(&_name));
}
