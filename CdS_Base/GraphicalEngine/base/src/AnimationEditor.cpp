#include "AnimationEditor.h"


AnimationEditor AnimationEditor::_sing = AnimationEditor();

AnimationEditor::AnimationEditor() : Widgetable("Animation_Editor", true),
_left("Left_Color"),
_right("Right_Color"),
_anim(NULL)
{
}


AnimationEditor::~AnimationEditor()
{
}

void AnimationEditor::newAnimation(void * v)
{
}

void AnimationEditor::loadAnimation(void * v)
{
}

void AnimationEditor::Build()
{
	_sing.build();
}

Widget * AnimationEditor::GetContainer()
{
	return _sing.getContainer();
}

Widgetable * AnimationEditor::GetSingleton()
{
	return &_sing;
}

void AnimationEditor::SetSavePath(const std::string & path)
{
	_sing._savePath = path;
}

void AnimationEditor::SetAnimation(Animation * animation)
{
	_sing._anim = animation;
	_sing._editor->setTexture(animation->getTexture(0));
}

std::string AnimationEditor::XMLName() const
{
	return "AnimationEditor";
}

void AnimationEditor::associate()
{
	WAssociateWidget("Container", new ContainerWidget(square_d(-1, -1, 1, 1)));

	WAssociateWidget("Container:RightList", new ListWidget());
	WAssociateWidget("Container:LeftContainer", new ContainerWidget(square_d(-1, 0, -1, 1)));

	_colorWid = new GetColorWidget();
	_colorWid->setSize(ge_pi(200, 200));
	_colorWid->setTexture(TextureDataBase::requestTexture("test"));
	_colorWid->setCallback(VoidedCallbackFunction(AnimationEditor, colorChanged), true);
	WAssociateWidget("Container:RightList:ColorTriangle", _colorWid);

	_editor = new SurfaceEditorWidget();
	WAssociateWidget("Container:LeftContainer:Editor", _editor);



	XMLAssociateSubBeacon("LeftColor", &_left);
	WAssociateBeacon("LeftColor", "Container:RightList");


	XMLAssociateSubBeacon("RightColor", &_right);
	WAssociateBeacon("RightColor", "Container:RightList");

}

void AnimationEditor::colorChanged(void * v)
{
	Uint8 button = (Uint8)v;
	switch (button) {
	case 1:
		_left.colorChanged(_colorWid->getColor());
		break;
	case 3:
		_right.colorChanged(_colorWid->getColor());
		break;
	}
}

AnimationEditor::EditorColor::EditorColor(const std::string & name) : Widgetable(name)
{
}

AnimationEditor::EditorColor::~EditorColor()
{
}

std::string AnimationEditor::EditorColor::XMLName() const
{
	return "EditorColor";
}

void AnimationEditor::EditorColor::associate()
{
	WAssociateWidget("Container", new ContainerWidget(square_d(-1, -1, -1, -1)));
	ColoredSquareWidget *csw = new ColoredSquareWidget();
	csw->setSize(ge_pi(40, 100));
	_color = csw->getColorInstance();

	_satuCol = ColorS(*_color);
	_satu = new GetColorWidget();
	_satu->setSize(ge_pi(40, 100));
	_satu->setCallback(VoidedCallbackFunction(EditorColor, satuColorChanged), true);
	_satu->setTexture(TextureDataBase::getTexture(Surface::getGradient(_satuCol)));

	WAssociateWidget("Container:Satudisplay", _satu);
	WAssociateWidget("Container:ColorList", new ListWidget());
	WAssociateWidget("Container:ColoredSquare", csw);

	WAssociateField("Red", "Container:ColorList", new XML::UInteger8(&_color->_r), SubBeaconLoadFunction(EditorColor, colorTextChanged));
	WAssociateField("Green", "Container:ColorList", new XML::UInteger8(&_color->_g), SubBeaconLoadFunction(EditorColor, colorTextChanged));
	WAssociateField("Blue", "Container:ColorList", new XML::UInteger8(&_color->_b), SubBeaconLoadFunction(EditorColor, colorTextChanged));
	WAssociateField("Satu", "Container:ColorList", new XML::Double(&_satuCol._s), SubBeaconLoadFunction(EditorColor, satuTextChanged));
	
}

void AnimationEditor::EditorColor::colorChanged(const Color & color)
{
	*_color = color;
	variableChanged(&_color->_r);
	variableChanged(&_color->_g);
	variableChanged(&_color->_b);
	colorTextChanged(NULL);
	_satu->setTexture(TextureDataBase::getTexture(Surface::getGradient(_satuCol)));
}

Color AnimationEditor::EditorColor::getColor() const
{
	return *_color;
}


void AnimationEditor::EditorColor::satuColorChanged(void * v)
{
	*_color = _satu->getColor();
	variableChanged(&_color->_r);
	variableChanged(&_color->_g);
	variableChanged(&_color->_b);
	colorTextChanged(v);
}

void AnimationEditor::EditorColor::colorTextChanged(void *v)
{
	_satuCol = ColorS(*_color);
	variableChanged(&_satuCol._s);
}

void AnimationEditor::EditorColor::satuTextChanged(void *v)
{
	*_color = _satuCol.getColor();
	variableChanged(&_color->_r);
	variableChanged(&_color->_g);
	variableChanged(&_color->_b);
}
