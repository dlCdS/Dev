#pragma once
#include "TextFieldWidget.h"
#include "ClickWidget.h"
#include "IconWidget.h"

class NamedIconWidget :
	public Widget
{
public:
	NamedIconWidget();
	virtual ~NamedIconWidget();


	void associateIcon(Animation *anim, XML::Field *field);
	void setTextSize(const ge_i &size);
	void updateText();
	void textCallback(std::string *str);

	virtual std::string XMLName() const;
	static std::string staticXMLName();

	void setForwardFunction(XML::VoidCallback forwardEdition);
	void setIconClicked(XML::VoidCallback iconClicked);

	virtual void updateContent();

protected:

	void onClickIcon(void *v);

	void noForward(void *v);
	virtual void postComputeRelative(const ge_pi &pen);

	XML::Field *_field;
	IconWidget *_icw;
	ge_i _textSize;
	XML::VoidCallback _forwardChange, _iconClicked;
};

