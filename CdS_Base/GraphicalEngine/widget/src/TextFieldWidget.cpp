#include "TextFieldWidget.h"



TextFieldWidget::TextFieldWidget() : 
	CallbackWidget(VoidedCallbackFunction(TextFieldWidget, triggerEdition), true),
	Widget(NULL, false, true, false, true, false, square_d(-1, -1, -1, -1)),
	_lib(LetterLib::getDefaultLib()),
	_forwardEdition(SubBeaconLoadFunction(TextFieldWidget, noForward))
{
}

TextFieldWidget::TextFieldWidget(const ge_i & textSize) : 
	CallbackWidget(VoidedCallbackFunction(TextFieldWidget, triggerEdition), true),
	Widget(NULL, false, true, false, true, false, square_d(-1, -1, -1, -1)),
_lib(LetterLib::getDefaultLib()),
_textSize(textSize),
_forwardEdition(SubBeaconLoadFunction(TextFieldWidget, noForward))
{
}

TextFieldWidget::TextFieldWidget(const ge_i & textSize, const std::string & msg) :
	CallbackWidget(VoidedCallbackFunction(TextFieldWidget, triggerEdition), true),
	Widget(NULL, false, true, false, true, false, square_d(-1, -1, -1, -1)),
	_lib(LetterLib::getDefaultLib()),
	_textSize(textSize),
	_forwardEdition(SubBeaconLoadFunction(TextFieldWidget, noForward))
{
	setText(msg);
}


TextFieldWidget::~TextFieldWidget()
{
}

void TextFieldWidget::setTextSize(const ge_i & size)
{
	if (size > 0){
		_textSize = size;
		sqdHasChanged();
	}
	else Log(ERROR, "Invalid text size ", size);
}

void TextFieldWidget::setText(const std::string & s)
{
	_str = s;
	_lib->split(s, _sentence, _wSize);
	sqdHasChanged();
}

void TextFieldWidget::addLetter(const char & c)
{
	_str += c;
	_lib->split(_str, _sentence, _wSize);
	sqdHasChanged();
}

void TextFieldWidget::popLetter()
{
	_str = _str.substr(0, _str.size() - 1);
	_lib->split(_str, _sentence, _wSize);
	sqdHasChanged();
}

void TextFieldWidget::postComputeRelative(const ge_pi & pen)
{
	ge_i l(0);
	ge_d space(_lib->getSpaceSize());
	_line.clear();
	_line.push_back(std::vector<int>());
	SDL_Rect pr = { 0, 0, 0, 0 };
	getFirstNotEmptyRelRect(pr);
	_rel.w = pr.w;
	for (int i = 0; i < _sentence.size(); i++) {
		l += _textSize*_wSize[i];
		if (l < _rel.w) {
			_line[_line.size() - 1].push_back(i);
		}
		else {
			_line.push_back(std::vector<int>());
			_line[_line.size() - 1].push_back(i);
			l = _textSize * _wSize[i];
		}
	}
	_rel.h = _line.size()*_textSize;
	if (_line.size() <= 1) {
		_rel.w = l;
	}
}

void TextFieldWidget::draw(SDL_Renderer *renderer)
{
	_color.setColor(renderer);
	SDL_RenderFillRect(renderer, &_abs);
	SDL_SetRenderDrawColor(renderer, 255, 0, 255, 255);
	SDL_RenderDrawRect(renderer, &_abs);
	
	SDL_Rect dst = _abs;
	dst.h = _textSize;
	LetterLib::Letter *l = NULL;
	for (auto line : _line) {
		dst.x = _abs.x;
		for (auto wi : line) {
			for (auto c : _sentence[wi]) {
				l = _lib->getLetter(c);
				dst.w = _lib->getLetter(c)->getWidth()*_textSize;
				SDL_RenderCopy(renderer, l->getTexture(), NULL, &dst);
				dst.x += dst.w;
			}
		}
		dst.y += _textSize;
	}
	if (_line.size() == 0) {
		dst.x = _abs.x;
		l = _lib->getLetter(' ');
		dst.w = _lib->getLetter(' ')->getWidth()*_textSize;
		SDL_RenderCopy(renderer, l->getTexture(), NULL, &dst);
	}
}

std::string TextFieldWidget::XMLName() const
{
	return staticXMLName();
}

std::string TextFieldWidget::staticXMLName()
{
	return "TextFieldWidget";
}

void TextFieldWidget::triggerEdition(void * v)
{
	new EditTextFieldEvent(this);
}

void TextFieldWidget::setEditable(const bool & disable)
{
	_clickable = !disable;
}

bool TextFieldWidget::letterExist(const char & c) const
{
	return _lib->letterExist(c);
}

void TextFieldWidget::setForwardFunction(XML::ReferenceLoader forwardEdition)
{
	_forwardEdition = forwardEdition;
}

void TextFieldWidget::doForward()
{
	_forwardEdition(&_str);
}



void TextFieldWidget::associate()
{
	Widget::associate();

	XMLAssociateField("str", new XML::String(&_str));

	XMLAddReference(LetterLib::staticXMLName(), SubBeaconGetReference(LetterLib, getLibName, _lib), SubBeaconLoadFunction(TextFieldWidget, loadLib));
}

void TextFieldWidget::loadLib(std::string *name)
{
	_lib = LetterLib::getLib(*name);
}

void TextFieldWidget::wordToLine()
{
	for (int i = 0; i < _wSize.size(); i++) {

	}
}

void TextFieldWidget::noForward(std::string * s)
{
}
