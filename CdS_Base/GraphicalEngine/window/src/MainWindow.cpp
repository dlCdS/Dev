#include "MainWindow.h"




MainWindow::MainWindow(const std::string &name) :
	Window(name),
	Parsable()
{
}

MainWindow::~MainWindow()
{
}


bool MainWindow::createWindow()
{
	Window::createWindow();

	LetterLib::setRenderer(_renderer);
	SDL_Surface *surface = SDL_GetWindowSurface(_window);
	TextureDataBase::setRenderer(_renderer, *surface->format);
	Widget::setRenderer(_renderer);
	LetterLib::getLib(DEFAULTLETTERLIB);
	SDL_FreeSurface(surface);

	return true;
}

SDL_Renderer * MainWindow::getRenderer()
{
	return _renderer;
}


SDL_Renderer * MainWindow::setRenderer()
{
	return SDL_CreateRenderer(_window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_TARGETTEXTURE);
}

void MainWindow::draw()
{
	SDL_SetRenderTarget(_renderer, NULL);
	SDL_SetRenderDrawBlendMode(_renderer, SDL_BLENDMODE_BLEND);
	SDL_SetRenderDrawColor(_renderer, 100, 100, 100, 0);
	SDL_RenderClear(_renderer);

	drawWidgets(_renderer);

	SDL_RenderPresent(_renderer);
}
