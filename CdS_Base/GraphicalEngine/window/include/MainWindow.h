#pragma once
#include "Window.h"

class MainWindow :
	public Window
{
public:
	MainWindow(const std::string &name);
	virtual ~MainWindow();

	virtual bool createWindow();
	SDL_Renderer *getRenderer();


protected:
	
	virtual SDL_Renderer *setRenderer();
	virtual void draw();
	
};

