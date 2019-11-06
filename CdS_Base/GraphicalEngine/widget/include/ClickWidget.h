#pragma once
#include "TextFieldWidget.h"

#define GLOW_TEXTURE_NAME

class ClickWidget :
	public CallbackWidget
{
public:
	ClickWidget();
	~ClickWidget();

	virtual void computeRelative(const ge_pi &pen);
	virtual void addWidget(Widget *widget);

	void setText(const std::string &text, const std::string &localText="");
	virtual void draw(SDL_Renderer *renderer);

	virtual std::string XMLName() const;
	static std::string staticXMLName();

	virtual void setColor(const Uint8 &r = DEFAULT_R, const Uint8 &g = DEFAULT_G, const Uint8 &b = DEFAULT_B);

	virtual void callback(void *v);
	void useTextAsParameter();
	void setGlowing();

protected:
	
	void onFocusGlow(void *v);
	void onLooseFocusGlow(void *v);
	std::string _text;
	bool _textAsInput, _glow;
	Uint8 _alpha;
};

