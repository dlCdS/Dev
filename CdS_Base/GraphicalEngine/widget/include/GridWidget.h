#pragma once
#include "Widget.h"
class GridWidget :
	public virtual Widget
{
public:
	GridWidget();
	GridWidget(const ge_pi &gridDim);
	~GridWidget();

	virtual void addWidget(Widget *widget);
	virtual void addWidget(Widget *widget, const ge_pi &size);
	virtual bool addWidgetAt(Widget *widget, const ge_pi &pos);
	virtual bool addWidgetAt(Widget *widget, const ge_pi &pos, const ge_pi &size);
	void setGridDim(const ge_pi &gridDim);
	ge_pi getGridDim() const;
	ge_i getGridSize() const;
	virtual void setGridAbstractSize(const ge_i &tilew, const ge_i &tileh = -1);

	virtual std::string XMLName() const;
	static std::string staticXMLName();


	virtual void scalingReduction();

protected:

	virtual void associate();
	ge_pi _gridDim, _cur;
};

