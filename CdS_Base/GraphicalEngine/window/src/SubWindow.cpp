#include "SubWindow.h"



SubWindow::SubWindow(const std::string &name) : Window(_name),
Parsable()
{
}


SubWindow::~SubWindow()
{
}

std::string SubWindow::XMLName() const
{
	return staticXMLName();
}

std::string SubWindow::staticXMLName()
{
	return "subWindow";
}

SDL_Renderer * SubWindow::setRenderer()
{
	return SDL_CreateRenderer(_window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_TARGETTEXTURE);
}

void SubWindow::draw()
{
	SDL_SetRenderTarget(_renderer, NULL);
	SDL_SetRenderDrawBlendMode(_renderer, SDL_BLENDMODE_BLEND);
	SDL_SetRenderDrawColor(_renderer, 100, 100, 100, 0);
	SDL_RenderClear(_renderer);

	drawWidgets(_renderer);

	SDL_RenderPresent(_renderer);
}
