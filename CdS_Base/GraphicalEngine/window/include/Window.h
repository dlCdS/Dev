#pragma once
#include "ListWidget.h"
#include "GridWidget.h"
#include "FitLineWidget.h"
#include "LetterLib.h"
#include "FieldWidget.h"
#include "ContainerWidget.h"
#include "LineWidget.h"
#include "ColoredSquareWidget.h"
#include "SurfaceEditorWidget.h"
#include "MoveGridWidget.h"
#include "PageWidget.h"
#include "NamedIconWidget.h"
#include "SetColourWidget.h"

class Window :
	public Widget, public XML::Compliant
{
public:
	Window(const std::string &name);
	~Window();

	void step();

	void setSize(const ge_pi &_size);
	virtual bool createWindow();
	
	void deleteWindow();

	Uint32 getId() const;

protected:

	virtual void drawWidgets(SDL_Renderer *renderer);

	virtual SDL_Renderer *setRenderer() = 0;

	XML::Parsable *provideListWidget(void *v);
	XML::Parsable *provideFitLineWidget(void *v);
	XML::Parsable *provideGridWidget(void *v);
	XML::Parsable *provideIconWidget(void *v);
	XML::Parsable *provideClickWidget(void *v);
	XML::Parsable *provideTextFieldWidget(void *v);

	virtual std::string XMLName() const;

	virtual void associate();
	virtual void draw() = 0;

	std::string _name;
	ge_pi _size;
	SDL_Window *_window;
	SDL_Renderer *_refRenderer, *_renderer;
	SDL_Texture *_target;
	Widget *_focus;

	static SDL_Window *_refWindow;
};

