#include "WaterColorModel.h"


WaterColorModel::WaterColorModel() : _pol(NULL), _data(NULL)
{
}


WaterColorModel::~WaterColorModel()
{
	destroyModel();
}

void WaterColorModel::setPolarityWidget(SetColourWidget * polw)
{
	_pol = polw;
	initWidget(_pol);
}

void WaterColorModel::setSize(const ge_pi & size, const ge_i &zoom)
{
	_size = size;
	_zoom = zoom;
}

void WaterColorModel::generate()
{
	destroyModel();
	_data = new ModelData*[_size.w];
	for (int i = 0; i < _size.w; i++){
		_data[i] = new ModelData[_size.h];
		for(int j=0; j<_size.h; j++)
			_data[i][j].initialize();
	}
	for (int i = 0; i < _size.w - 1; i++) {
		for (int j = 0; j < _size.h - 1; j++) {
			_interface.push_back(Interface(&_data[i][j], &_data[i + 1][j], { 1.0, 0 }));
			_interface.push_back(Interface(&_data[i][j], &_data[i][j + 1], { 0, 1.0 }));
		}
	}
}

void WaterColorModel::cycle()
{
	Clocks::start("cycle");
	Clocks::start("upInterface");
	updateInterface();
	Clocks::stop("upInterface");
	Clocks::start("upData");
	updateData();
	Clocks::stop("upData");
	Clocks::start("drawPol");
	drawPolariy();
	Clocks::stop("drawPol");
	Clocks::stop("cycle");
}

void WaterColorModel::colorDrop(const int & radius, const ge_d &quantity)
{
	for(int i=-radius; i<radius; i++)
		for (int j = -radius; j < radius; j++) {
			if (sqrt(i*i + j * j) <= radius) {
				_data[_size.w / 2 + i][_size.h / 2 + j].setColor({ 0, 0, 255 }, quantity);
			}
		}
}

void WaterColorModel::updateData()
{
	for(int i=0; i<_size.w;i++)
		for (int j = 0; j < _size.h; j++) {
			_data[i][j].update();
		}
}

void WaterColorModel::updateInterface()
{
	for (auto i = _interface.begin(); i != _interface.end(); ++i) {
		i->updateData();
	}
}

void WaterColorModel::drawPolariy()
{
	Uint8 b;
	for(int i=0; i<_size.w; i++)
		for (int j = 0; j < _size.h; j++) {
			if (_data[i][j].w > 0.1)
				b = 255;
			else b = 0;
			for(int k=0;k<_zoom;k++)
				for(int l=0;l<_zoom;l++)
					_pol->setPixelColor(_zoom*i+k, _zoom*j+l, { (Uint8)((_data[i][j].p.x + 1) * 255 / 2), (Uint8)((_data[i][j].p.y + 1) * 255 / 2) , b });
		}
	_pol->updateTexture();
}

void WaterColorModel::destroyModel()
{
	if (_data != NULL) {
		for (int i = 0; i < _size.w; i++)
			delete[] _data[i];
		delete _data;
	}
	_data = NULL;
	_interface.clear();
}

void WaterColorModel::initWidget(SetColourWidget * widget)
{
	widget->setSize({ _size.w*_zoom, _size.h*_zoom });
	widget->generateSurface();
}


WatercolorModel::WatercolorModel()
{
}


WatercolorModel::~WatercolorModel()
{
}
