#include "FieldWidget.h"



FieldWidget::FieldWidget() : 
	Widget(NULL, false, false, false, false, false, square_d(-1, -1, -1, -1)),
	_forwardChange(VoidedCallbackFunction(FieldWidget, noForward)),
	_field(NULL)
{
}


FieldWidget::~FieldWidget()
{
	delete _field;
}

void FieldWidget::associateField(const std::string &fname, XML::Field * field)
{
	_fieldName = fname;
	_field = field;
	addWidget(new TextFieldWidget(_textSize));
	TextFieldWidget *tf = new TextFieldWidget(_textSize);
	tf->setForwardFunction(SubBeaconLoadFunction(FieldWidget, callback));
	tf->setEditable();
	addWidget(tf);
	updateText();
}

void FieldWidget::setTextSize(const ge_i & size)
{
	_textSize = size;
}

void FieldWidget::updateText()
{
	dynamic_cast<TextFieldWidget*>(_child[0])->setText(_fieldName);
	dynamic_cast<TextFieldWidget*>(_child[1])->setText(_field->get());
}

void FieldWidget::callback(std::string * str)
{
	_field->set(*str);
	updateText();
	_forwardChange(NULL);
}

std::string FieldWidget::XMLName() const
{
	return staticXMLName();
}

std::string FieldWidget::staticXMLName()
{
	return "FieldWidget";
}

void FieldWidget::setForwardFunction(XML::VoidCallback forwardEdition)
{
	_forwardChange = forwardEdition;
}

void FieldWidget::updateContent()
{
	updateText();
}

void FieldWidget::noForward(void * v)
{
}

void FieldWidget::postComputeRelative(const ge_pi & pen)
{
	if (_sqd.dim.w < 0) {
		_rel.w = 0;
		for (auto c : _child)
			_rel.w += c->getRelativeRect().w;
	}
	if (_sqd.dim.h < 0)
		_rel.h = _rdcorner.h;
}
