#pragma once
#include "TextFieldWidget.h"
class FieldWidget :
	public Widget
{
public:
	FieldWidget();
	~FieldWidget();

	void associateField(const std::string &fname,  XML::Field *field);
	void setTextSize(const ge_i &size);
	void updateText();
	void callback(std::string *str);

	virtual std::string XMLName() const;
	static std::string staticXMLName();

	void setForwardFunction(XML::VoidCallback forwardEdition);


	virtual void updateContent();

protected:

	void noForward(void *v);
	virtual void postComputeRelative(const ge_pi &pen);

	XML::Field *_field;
	std::string _fieldName;
	ge_i _textSize;
	XML::VoidCallback _forwardChange;

};

