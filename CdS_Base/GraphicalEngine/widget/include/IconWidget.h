#pragma once
#include "Widget.h"

#define DEFAULT_ICON_SIZE 40
class IconWidget :
	public Widget
{
public:
	IconWidget();
	IconWidget(Widget *parent, Animation *animation);
	~IconWidget();

	virtual void draw(SDL_Renderer *renderer);
	void setSize(const ge_pi &size);
	void setAnimation(Animation *animation);

	std::string XMLName() const;
	static std::string staticXMLName();

	Animation *getAnimation();

protected:
	virtual void postComputeRelative(const ge_pi &pen);
	Parsable *checkAnimationUnicity(void *v);
	Parsable *getAnimation(void *v);
	void loadAnimation(std::string *s);
	virtual void computeAbsolute(const SDL_Rect &container, const SDL_Rect &seen, const SDL_Rect &offset);
	SDL_Rect _srcrect;


	virtual void associate();
	ge_pi _size;
	Animation *_animation;
};

