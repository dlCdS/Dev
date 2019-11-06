#pragma once
#include "CallbackWidget.h"
#include "LetterLib.h"

class TextFieldWidget :
	public CallbackWidget
{
public:
	TextFieldWidget();
	TextFieldWidget(const ge_i & textSize);
	TextFieldWidget(const ge_i & textSize, const std::string &msg);
	~TextFieldWidget();

	void setTextSize(const ge_i &size);
	void setText(const std::string &s);
	void addLetter(const char &c);
	void popLetter();

	virtual void postComputeRelative(const ge_pi &pen);
	virtual void draw(SDL_Renderer *renderer);

	virtual std::string XMLName() const;
	static std::string staticXMLName();

	void triggerEdition(void *v);
	void setEditable(const bool &disable = false);
	bool letterExist(const char &c) const;
	void setForwardFunction(XML::ReferenceLoader forwardEdition);
	void doForward();
	
protected:
	virtual void associate();

	void loadLib(std::string *name);

private:

	void wordToLine();
	void noForward(std::string *s);

	SDL_Texture *_displayable;
	LetterLib *_lib;
	std::string _str;
	ge_i _textSize;
	std::vector<std::string > _sentence;
	std::vector<std::vector<int> > _line;
	std::vector<ge_d > _wSize;
	XML::ReferenceLoader _forwardEdition;
};

