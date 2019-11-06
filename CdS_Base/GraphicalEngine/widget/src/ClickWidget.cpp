#include "ClickWidget.h"



ClickWidget::ClickWidget() : 
	CallbackWidget(),
	Widget(NULL, 0, 0, 1, 0, 0, square_d(-1, -1, -1, -1)),
	_textAsInput(false)
{
}


ClickWidget::~ClickWidget()
{
}

void ClickWidget::computeRelative(const ge_pi & pen)
{
	commonComputeRelative(pen);
}

void ClickWidget::addWidget(Widget * widget)
{
	if (_child.size() == 0) {
		Widget::addWidget(widget);
		widget->setRelativeProportion(square_d(-1, -1, -1, -1));
	}
	else _child.push_back(widget);
}

void ClickWidget::setText(const std::string & text, const std::string &localText)
{
	if (_child.size() == 0){
		TextFieldWidget *tf = new TextFieldWidget(30);
		tf->setText(text);
		addWidget(tf);
	}
	if(localText=="")
		_text = text;
	else
		_text = localText;
}

void ClickWidget::draw(SDL_Renderer * renderer)
{
	_color.setColor(renderer, _alpha);
	SDL_RenderFillRect(renderer, &_seen);
}


std::string ClickWidget::XMLName() const
{
	return staticXMLName();
}

std::string ClickWidget::staticXMLName()
{
	return "clickWidget";
}

void ClickWidget::setColor(const Uint8 & r, const Uint8 & g, const Uint8 & b)
{
	Widget::setColor(r, g, b);
	if (_child.size() > 0)
		_child[0]->setColor(r, g, b);
}

void ClickWidget::callback(void * v)
{
	if (_textAsInput)
		_callback((void*)&_text);
	else _callback(v);
}

void ClickWidget::useTextAsParameter()
{
	_textAsInput = true;
}

void ClickWidget::setGlowing()
{
	_glow = true;
	_color = { 255, 255, 255 };
	_onFocus = VoidedCallbackFunction(ClickWidget, onFocusGlow);
	_looseFocus = VoidedCallbackFunction(ClickWidget, onLooseFocusGlow);
}

void ClickWidget::onFocusGlow(void * v)
{
	_alpha = 150;
}

void ClickWidget::onLooseFocusGlow(void * v)
{
	_alpha = 0;
}

