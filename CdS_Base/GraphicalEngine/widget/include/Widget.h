#pragma once

#include <vector>
#include <math.h>
#include "PublicSdlEventServer.h"
#include "LoopbackEvent.h"
#include "AnimationDatabase.h"


class WidgetLayer;

class Widget :
	public virtual XML::Parsable
{
public:
	Widget();
	Widget(Widget *parent);
	Widget(Widget *parent, const bool scrollable, const bool autoFeedLine, const bool clickable, const bool noChild, const bool &onePerLine);
	Widget(Widget *parent, const bool scrollable, const bool autoFeedLine, const bool clickable, const bool noChild, const bool &onePerLine, const square_d &dim);
	virtual ~Widget();

	void computeRelative(const ge_pi &pen);
	virtual void draw(SDL_Renderer *renderer);
	virtual void freeAll();
	virtual void addWidget(Widget *widget);
	void setParent(Widget *parent);
	void addLayer();
	void setRelativeProportion(const square_d &sqd); // <0 means as much as needed
	void setRelativeSquare(const SDL_Rect &rel); // <0 means as much as needed
	void setAbstractSquare(const SDL_Rect &abs); // <0 means as much as needed
	void setAbstractPos(const ge_i &x, const ge_i &y);

	virtual bool getWidgetAt(const SDL_Point &pos, std::list<Widget*> &list);

	SDL_Rect getAbsRect() const;
	SDL_Rect getRelativeRect() const;
	void getFirstNotEmptyRelRect(SDL_Rect &rect);

	static bool RectEqual(const SDL_Rect &r1, const SDL_Rect &r2);

	virtual void updateContent();
	virtual void setColor(const Uint8 &r= DEFAULT_R, const Uint8 &g= DEFAULT_G, const Uint8 &b= DEFAULT_B);
	Color *getColorInstance();

	void setShared();
	virtual void drawWidgets(SDL_Renderer *renderer);
	const bool &isShared() const;
	void sqdHasChanged(const bool &propagateup = true, const bool &propagatedown = false);

	static void setRenderer(SDL_Renderer* renderer);

protected:

	static SDL_Renderer* Renderer;

	void relHasChanged(const bool &propagateup=true, const bool &propagatedown = false);
	Parsable *XMLAddWidgetLayer(void *v);
	virtual void associate();
	virtual void computeAbsolute(const SDL_Rect &container, const SDL_Rect &seen, const SDL_Rect &offset);

	void removeChild(Widget *w);

	void computeChild();
	void commonComputeRelative(const ge_pi &pen);
	virtual void postComputeRelative(const ge_pi &pen);
	ge_pi getPen() const;

	SDL_Rect _abs, _rel, _abstract, _seen;
	square_d _sqd;
	Widget *_parent;
	ge_pi _pen, _rdcorner;
	std::vector<Widget *> _child;
	std::vector<WidgetLayer *> _layers;
	bool _scrollable, _autoFeedLine, _clickable, _noChild, _onePerLine,
		_relHasChanged, _sqdHasChanged, _shared;
	Color _color;


	XML::DoubleSquare _xmlsqd;

};

class WidgetLayer : public XML::Parsable {
public:
	WidgetLayer(Widget *parent) : _parent(parent) {}

	Parsable *addWidget(XML::Parsable *p) {
		Widget *w = dynamic_cast<Widget *>(p);
		w->setParent(_parent);
		_parent->addWidget(w);
		return w;
	}
	static std::string staticXMLName() { return "layer"; }
	std::string XMLName() const {	return staticXMLName();	}

	virtual void associate() {
		XMLAssociatePolymorph("layer",  _wid.begin(), _wid.end(), SubBeaconLoadFunction(WidgetLayer, addWidget));
	}
	void add(Widget *w) {
		_wid.push_back(w);
	}
	std::list<Widget*> _wid;
	Widget *_parent;
};