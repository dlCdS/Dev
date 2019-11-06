#include "GridWidget.h"



GridWidget::GridWidget() : 
	Widget(),
	_cur(ge_pi(0, 0))
{
}

GridWidget::GridWidget(const ge_pi & gridDim) :
	Widget(NULL, 0, 0, 0, 0, 0, square_d(-1, -1, -1, -1)),
	_gridDim(gridDim),
	_cur(ge_pi(0, 0))
{
}

GridWidget::~GridWidget()
{
}

void GridWidget::addWidget(Widget * widget)
{
	if (_layers.size() == 0)
		addLayer();
	addWidgetAt(widget, _cur);
}

void GridWidget::addWidget(Widget * widget, const ge_pi & size)
{
	if (_layers.size() == 0)
		addLayer();
	addWidgetAt(widget, _cur, size);
}

bool GridWidget::addWidgetAt(Widget * widget, const ge_pi & pos)
{
	_child.push_back(widget);
	widget->setParent(this);
	if (!_noChild && _cur.w < _gridDim.w && _cur.h < _gridDim.h) {
		_layers[_layers.size() - 1]->add(widget);
		widget->setRelativeProportion(square_d(
			1.0 * _cur.w / _gridDim.w,
			1.0 * _cur.h / _gridDim.h,
			1.0 / _gridDim.w,
			1.0 / _gridDim.h));
		if (++_cur.w >= _gridDim.w)
			_cur = { 0, _cur.h + 1 };
		return true;
	} else return false;
}

bool GridWidget::addWidgetAt(Widget * widget, const ge_pi & pos, const ge_pi & size)
{
	_child.push_back(widget);
	widget->setParent(this);
	if (!_noChild && _cur.w < _gridDim.w && _cur.h < _gridDim.h) {
		_layers[_layers.size() - 1]->add(widget);
		widget->setRelativeProportion(square_d(
			1.0 * _cur.w / _gridDim.w,
			1.0 * _cur.h / _gridDim.h,
			(ge_d) size.w / _gridDim.w,
			(ge_d) size.h / _gridDim.h));
		_cur.w += size.w;
		if (_cur.w >= _gridDim.w){
			_cur = { 0, _cur.h + size.h };
		}
		return true;
	}
	else return false;
}

void GridWidget::setGridDim(const ge_pi & gridDim)
{
	_cur = { 0, 0 };
	_gridDim = gridDim;
}

ge_pi GridWidget::getGridDim() const
{
	return _gridDim;
}

ge_i GridWidget::getGridSize() const
{
	return _gridDim.w*_gridDim.h;
}

void GridWidget::setGridAbstractSize(const ge_i & tilew, const ge_i & tileh)
{
	_abstract.w = tilew*_gridDim.w;
	if (tileh < 0)
		_abstract.h = tilew * _gridDim.h;
	else _abstract.h = tileh * _gridDim.h;
}

std::string GridWidget::XMLName() const
{
	return staticXMLName();
}

std::string GridWidget::staticXMLName()
{
	return "gridWidget";
}

void GridWidget::scalingReduction()
{
	if (_gridDim.h >= 4 && _gridDim.w >= 4) {
		ge_pi uple = { _gridDim.w / 2, _gridDim.h / 2 },
			upri = { _gridDim.w - uple.w,  uple.h },
			dole = { uple.w, _gridDim.h - uple.h },
			dori = { _gridDim.w - uple.w, _gridDim.h - uple.h };
		ge_i pos, x, y;
		GridWidget *grid[2][2] = { {new GridWidget(uple), new GridWidget(dole)}, { new GridWidget(upri), new GridWidget(dori)} };
		for (int j = 0; j < _gridDim.h; j++)
		for (int i = 0; i < _gridDim.w; i++)  {
				pos = i + j * _gridDim.w;
				if (pos < _child.size()) {
					if (i < uple.w)
						x = 0;
					else x = 1;
					if (j < uple.h)
						y = 0;
					else y = 1;
					grid[x][y]->addWidget(_child[pos]);
				}
			}
		_child.clear();
		freeAll();
		setGridDim(ge_pi(_gridDim.w, _gridDim.h));
			for (int j = 0; j < 2; j++)
				for (int i = 0; i < 2; i++) {
			addWidget(grid[i][j], ge_pi(_gridDim.w / 2 + i*(_gridDim.w - _gridDim.w/2 - _gridDim.w/2), _gridDim.h / 2 + j* (_gridDim.h - _gridDim.h / 2 - _gridDim.h / 2)));
			grid[i][j]->scalingReduction();
		}

	}
}

void GridWidget::associate()
{
	Widget::associate();
	XMLAssociateField("gridWidth", new XML::Integer(&_gridDim.w));
	XMLAssociateField("gridHeigh", new XML::Integer(&_gridDim.h));
}
