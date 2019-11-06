#pragma once

#include "TextureDataBase.h"
#include <fstream>
#include <unordered_set>
#include <XMLParsable.h>

#define DEFAULTANIMATION "..\\widgets\\anim\\defaultWidget.anim"

class Animation : public XML::Parsable
{
public:
	Animation();
	Animation(SDL_Texture *texture);
	~Animation();

	static Animation* getDefaultAnimation();
	static void updateTime();
	static unsigned long &getTime();
	SDL_Texture *getTexture(const unsigned long &time=_time) const;
	void load(const char *filename, const std::string &newname="");
	void save(const char* filename);
	unsigned long getDuration() const;
	void addFrame();
	void removeFrame();
	void addTexture(const std::string & filename);
	void addTexture(SDL_Texture *t);
	bool textureFirstOccurence(const unsigned long &time = _time) const;
	int getNumberOfTexture() const;

	virtual std::string getFilename();

	std::string XMLName() const;
	static std::string staticXMLName();

protected:

	virtual void associate();

private:
	std::vector<SDL_Texture*> _t;

	std::string _name;

	static unsigned long _time;
};

