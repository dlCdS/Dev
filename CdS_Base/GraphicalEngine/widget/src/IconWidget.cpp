#include "IconWidget.h"




IconWidget::IconWidget() : Widget(NULL, false, false, false, true, false, square_d(0, 0, -1, -1))
{
}

IconWidget::IconWidget(Widget * parent, Animation *animation) : Widget(parent, false, false, false, 
	true, false, square_d(0, 0, -1, -1)),
_animation(animation)
{
}

IconWidget::~IconWidget()
{
}

void IconWidget::postComputeRelative(const ge_pi & pen)
{
	if (_sqd.dim.w < 0)
		_rel.w = _size.w;
	if (_sqd.dim.h < 0)
		_rel.h = _size.h;
}

void IconWidget::draw(SDL_Renderer * renderer)
{
	SDL_SetRenderDrawColor(renderer, 255, 0, 255, 255);
	SDL_RenderDrawRect(renderer, &_seen);
	SDL_RenderCopy(renderer, _animation->getTexture(), &_srcrect, &_seen);
}

void IconWidget::setSize(const ge_pi & size)
{
	_size = size;
}

void IconWidget::setAnimation(Animation * animation)
{
	_animation = animation;
}


std::string IconWidget::XMLName() const
{
	return staticXMLName();
}

std::string IconWidget::staticXMLName()
{
	return "IconWidget";
}

Animation * IconWidget::getAnimation()
{
	return _animation;
}

XML::Parsable * IconWidget::checkAnimationUnicity(void * v)
{
	std::string file = _animation->getFilename();
	delete _animation;
	_animation = AnimationDataBase::requestAnimation(file);
	return nullptr;
}

XML::Parsable * IconWidget::getAnimation(void * v)
{
	_animation = new Animation();
	return _animation;
}

void IconWidget::loadAnimation(std::string * s)
{
	_animation = AnimationDataBase::requestAnimation(*s);
}

void IconWidget::computeAbsolute(const SDL_Rect & container, const SDL_Rect & seen, const SDL_Rect & offset)
{
	Widget::computeAbsolute(container, seen, offset);
	SDL_Rect textrect;
	SDL_QueryTexture(_animation->getTexture(), NULL, NULL, &textrect.w, &textrect.h);
	square_d reldest = {
		(ge_d)(_seen.x - _abs.x) / _abs.w,
		(ge_d)(_seen.y - _abs.y) / _abs.h,
		(ge_d)_seen.w / _abs.w,
		(ge_d)_seen.h / _abs.h
	};
	_srcrect = {
		(ge_i) (reldest.pos.w * textrect.w),
		(ge_i)(reldest.pos.h * textrect.h),
		(ge_i)(reldest.dim.w * textrect.w),
		(ge_i)(reldest.dim.h * textrect.h)
	};
}

void IconWidget::associate()
{
	Widget::associate();
	XMLAssociateField("width", new XML::Integer(&_size.w));
	XMLAssociateField("heigh", new XML::Integer(&_size.h));

	XMLAddReference("animation", SubBeaconGetReference(Animation, getFilename, _animation), SubBeaconLoadFunction(IconWidget, loadAnimation));
}
