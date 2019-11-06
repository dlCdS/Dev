#pragma once
#include "Window.h"
class SubWindow :
	public Window
{
public:
	SubWindow(const std::string &name);
	~SubWindow();


	virtual std::string XMLName() const;
	static std::string staticXMLName();

protected:

	virtual SDL_Renderer *setRenderer();
	virtual void draw();
};

