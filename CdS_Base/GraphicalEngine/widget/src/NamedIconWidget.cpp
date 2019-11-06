#include "NamedIconWidget.h"


NamedIconWidget::NamedIconWidget() :
	Widget(NULL, false, false, false, false, false, square_d(-1, -1, -1, -1)),
	_forwardChange(VoidedCallbackFunction(NamedIconWidget, noForward)),
	_iconClicked(VoidedCallbackFunction(NamedIconWidget, noForward)),
	_field(NULL),
	_icw(NULL)
{
}


NamedIconWidget::~NamedIconWidget()
{
	delete _field;
}

void NamedIconWidget::associateIcon(Animation * anim, XML::Field * field)
{
	_field = field;
	ClickWidget *cw = new ClickWidget();
	_icw = new IconWidget();
	_icw->setAnimation(anim);
	_icw->setSize({ DEFAULT_ICON_SIZE , DEFAULT_ICON_SIZE });
	cw->addWidget(_icw);
	cw->setCallback(VoidedCallbackFunction(NamedIconWidget, onClickIcon), true);
	cw->setGlowing();
	addWidget(cw);
	TextFieldWidget *tf = new TextFieldWidget(_textSize);
	tf->setForwardFunction(SubBeaconLoadFunction(NamedIconWidget, textCallback));
	tf->setEditable();
	addWidget(tf);
	updateText();
}

void NamedIconWidget::setTextSize(const ge_i & size)
{
	_textSize = size;
}

void NamedIconWidget::updateText()
{
	dynamic_cast<TextFieldWidget*>(_child[1])->setText(_field->get());
}

void NamedIconWidget::textCallback(std::string * str)
{
	_field->set(*str);
	updateText();
	_forwardChange(NULL);
}

std::string NamedIconWidget::XMLName() const
{
	return staticXMLName();
}

std::string NamedIconWidget::staticXMLName()
{
	return "NamedIcon";
}

void NamedIconWidget::setForwardFunction(XML::VoidCallback forwardEdition)
{
	_forwardChange = forwardEdition;
}

void NamedIconWidget::setIconClicked(XML::VoidCallback iconClicked)
{
	_iconClicked = iconClicked;
}

void NamedIconWidget::updateContent()
{
	updateText();
}

void NamedIconWidget::onClickIcon(void * v)
{
	if (_icw != NULL)
		_iconClicked((void*)_icw->getAnimation());
	else _iconClicked(NULL);
}

void NamedIconWidget::noForward(void * v)
{
}

void NamedIconWidget::postComputeRelative(const ge_pi & pen)
{
	if (_sqd.dim.w < 0) {
		_rel.w = 0;
		for (auto c : _child)
			_rel.w += c->getRelativeRect().w;
	}
	if (_sqd.dim.h < 0)
		_rel.h = _rdcorner.h;
}
