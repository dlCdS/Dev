#include "Window.h"



Window::Window(const std::string & name) : 
	Widget(NULL, false, false, false, false, false, square_d(-1, -1, -1, -1)),
	Compliant(name),
	Parsable(), 
	_name(name),
	_size(DEFAULT_WINDOW_SIZE),
	_window(NULL)
{
}

Window::~Window()
{
	deleteWindow();
}

void Window::step()
{
	draw();
}

void Window::setSize(const ge_pi & size)
{
	_size = size;
	_sqd = { 0, 0, 1, 1 };
	_abs = { 0, 0, (int)_size.w, (int)_size.h };
	_rel = _abs;
}

bool Window::createWindow()
{
	_window = SDL_CreateWindow(_name.c_str(), SDL_WINDOWPOS_UNDEFINED,
		SDL_WINDOWPOS_UNDEFINED,
		_size.w,
		_size.h,
		SDL_WINDOW_SHOWN);

	if (!_window)
	{
		Log(ERROR, "Erreur de création de la fenêtre: ", SDL_GetError());
		return false;
	}

	_renderer = setRenderer();
	SDL_SetRenderDrawBlendMode(_renderer, SDL_BLENDMODE_BLEND);
	_target = SDL_CreateTexture(_renderer, SDL_PIXELFORMAT_RGBA32, SDL_TEXTUREACCESS_TARGET, _size.w, _size.h);


	Log(LINFO, "Window created: ", _name, " size ", _size.w, "x", _size.h);
	new WindowCreatedEvent("Window has been created: "+_name);

	return true;
}

void Window::deleteWindow()
{
	SDL_DestroyRenderer(_renderer);
	SDL_DestroyWindow(_window);
}

Uint32 Window::getId() const
{
	return SDL_GetWindowID(_window);
}

void Window::drawWidgets(SDL_Renderer * renderer)
{
	if (_sqdHasChanged)
		computeChild();
	if (_relHasChanged)
		computeAbsolute(_abs, _abs, _abstract);
	if (_seen.h > 0 && _seen.w > 0) {
		for (auto l : _layers)
			for (auto w : l->_wid)
				w->drawWidgets(renderer);
		Widget::draw(renderer);
	}
}

XML::Parsable * Window::provideListWidget(void * v)
{
	return new ListWidget();
}

XML::Parsable * Window::provideFitLineWidget(void * v)
{
	return new FitLineWidget();
}

XML::Parsable * Window::provideGridWidget(void * v)
{
	return new GridWidget();
}

XML::Parsable * Window::provideIconWidget(void * v)
{
	return new IconWidget();
}

XML::Parsable * Window::provideClickWidget(void * v)
{
	return new ClickWidget();
}

XML::Parsable * Window::provideTextFieldWidget(void * v)
{
	return new TextFieldWidget();
}

std::string Window::XMLName() const
{
	return _name;
}

void Window::associate()
{
	Widget::associate();
	XMLAssociateField("width", new XML::Integer(&_size.w));
	XMLAssociateField("heigh", new XML::Integer(&_size.h));
	XMLAssociateField("name", new XML::String(&_name));


	XMLAddInstanceProvider(ListWidget::staticXMLName(), SubBeaconLoadFunction(Window, provideListWidget));
	XMLAddInstanceProvider(GridWidget::staticXMLName(), SubBeaconLoadFunction(Window, provideGridWidget));
	XMLAddInstanceProvider(IconWidget::staticXMLName(), SubBeaconLoadFunction(Window, provideIconWidget));
	XMLAddInstanceProvider(FitLineWidget::staticXMLName(), SubBeaconLoadFunction(Window, provideFitLineWidget));
	XMLAddInstanceProvider(ClickWidget::staticXMLName(), SubBeaconLoadFunction(Window, provideClickWidget));
	XMLAddInstanceProvider(TextFieldWidget::staticXMLName(), SubBeaconLoadFunction(Window, provideTextFieldWidget));
}
