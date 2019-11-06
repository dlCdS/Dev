#pragma once
#include "Widget.h"
class ColoredSquareWidget :
	public Widget
{
public:
	ColoredSquareWidget();
	~ColoredSquareWidget();

	void setSize(const ge_pi &size);
	virtual void draw(SDL_Renderer *renderer);

	std::string XMLName() const;
	static std::string staticXMLName();


protected:
	virtual void postComputeRelative(const ge_pi &pen);


	virtual void associate();
	ge_pi _size;
};

