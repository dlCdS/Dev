#include "Weather.h"

MeteoModel MeteoModel::_sing = MeteoModel();
ge_d Weather::_am = 0.0;

Weather::Weather() : XML::Parsable(),
_s(0.0),
_w(0.0),
_m(0.0),
_wir(0.0),
_war(0.0),
_a(0.0),
_ar(0.0),
_rain(false),
_sea(false)
{
}


Weather::~Weather()
{
}

void Weather::heatTransfert()
{
	ge_d dh(HEAT_TRANSFERT_RATE*(_mg._t - _ma._t));
	_mg._t -= dh;
	_ma._t += dh;
}

void Weather::waterTransfert()
{
	// snow <-> water transformation
	if (_mg._t < SNOW_WATER_TRANSFORM_TRIGGER) {
		_s += SNOW_WATER_TRANSFORM_RATE * _mg._w;
		_mg._w -= SNOW_WATER_TRANSFORM_RATE * _mg._w;
	}
	else if (_mg._t < SNOW_WATER_TRANSFORM_TRIGGER && _s > ALMOST_ZERO) {
		_mg._w += SNOW_WATER_TRANSFORM_RATE * _s;
		_s -= SNOW_WATER_TRANSFORM_RATE * _s;
	}

	if (_ma._w > 1.0) {
		_rain = true;
		_m += RAIN_TRANSFERT_RATE * _ma._w;
		_ma._w -= _m;
		if (_m > 1.0) {
			_mg._w += (_m - 1.0);
			_m = 1.0;
		}
	}
	else {
		_rain = false;
		_m += MOISTURE_TRANSFERT_RATE * _mg._w * _mg._t;
		_mg._w -= _m;
		if (_m > 1.0) {
			_ma._w += (_m - 1.0);
			_m = 1.0;
		}
	}
}

void Weather::sunHeat(const ge_d & light)
{
	if (_ma._w < 1.0) {
		_mg._t += (1.0 - _ma._w) * light * HEAT_TRANSFERT_RATE * (1.0 - _mg._t);
	}
}

void Weather::altitudeCooling()
{
	_ma._t -= _ma._t * _ar / _am * ALTITUDE_HEAT_LOSS;
}

void Weather::loop(const ge_d &light)
{
	heatTransfert();
	waterTransfert();
	sunHeat(light);
	if(!_sea){
		altitudeCooling();
	}
	else {
		_mg._w = 1.0;
	}
}

void Weather::associate()
{
}

std::string Weather::XMLName() const
{
	return "Weather";
}

MeteoModel::~MeteoModel()
{
	for(int i=0; i<_s.w; i++)
		delete[] _m[i];
	delete[] _m;
}

void MeteoModel::setSize(const ge_pi & size)
{
	_sing._s = size;
}

void MeteoModel::generate()
{
	if(!_sing._locked){
		_sing._locked = true;
		_sing._m = new Weather*[_sing._s.w];
		for (int i = 0; i < _sing._s.w; i++) {
			_sing._m[i] = new Weather[_sing._s.h];
		}
		for (int i = 0; i < _sing._s.w; i++) 
			for (int j = 0; j < _sing._s.h; j++) {
				if (j < _sing._s.h - 1)
					_sing._i.push_back(Interface(&_sing._m[i][j], &_sing._m[i][j + 1]));
				if (i < _sing._s.w - 1)
					_sing._i.push_back(Interface(&_sing._m[i][j], &_sing._m[i + 1][j]));
		}
	}

}

void MeteoModel::test(const ge_pi & size, const int &ntest)
{
	time_t time(0), tw(0), ti(0);
	setSize(size);
	generate();
	for (int i = 0; i < ntest; i++) {
		Clocks::start("MeteoModel::i");
		_sing.interfaceLoop();
		Clocks::stop("MeteoModel::i");
	}
	Clocks::report();

}

std::string MeteoModel::XMLName() const
{
	return "MeteoModel";
}

void MeteoModel::associate()
{
}

void MeteoModel::step()
{
	// loop on weather tiles
	
	// loop on interfaces
	for (auto i : _i) {
		i.getStiff();
		i.getWind();
		i.airMove();
		i.waterMove();
	}

}

void MeteoModel::weatherLoop()
{
	for (int i = 0; i < _sing._s.w; i++)
		for (int j = 0; j < _sing._s.h; j++) {
			_m[i][j].heatTransfert();
			_m[i][j].waterTransfert();
			_m[i][j].sunHeat(_light);
			_m[i][j].altitudeCooling();
		}
}

void MeteoModel::interfaceLoop()
{
	Clocks::start("MeteoModel::i:getStiff");
	for (int i = _offset; i < _i.size(); i+=_tempo) {
		_i[i].getStiff();
	}
	Clocks::stop("MeteoModel::i:getStiff");

	Clocks::start("MeteoModel::i:getWind");
	for (int i = _offset; i < _i.size(); i += _tempo) {
		_i[i].getWind();
	}
	Clocks::stop("MeteoModel::i:getWind");

	Clocks::start("MeteoModel::i:airMove");
	for (int i = _offset; i < _i.size(); i += _tempo) {
		_i[i].airMove();
	}
	Clocks::stop("MeteoModel::i:airMove");

	Clocks::start("MeteoModel::i:waterMove");
	for (int i = _offset; i < _i.size(); i += _tempo) {
		_i[i].waterMove();
	}
	Clocks::stop("MeteoModel::i:waterMove");

	Clocks::start("MeteoModel::i:weather");
	for (int i = _offset; i < _i.size(); i += _tempo) {
		_i[i]._w1->loop(_light);
		_i[i]._w2->loop(_light);
	}
	Clocks::stop("MeteoModel::i:weather");
	_offset = (_offset + 1) % _tempo;
}

MeteoModel::MeteoModel() : XML::Parsable(),
_m(NULL),
_light(0.5),
_locked(false),
_tempo(100),
_offset(0)
{
}

ge_d MeteoModel::Interface::realStiff() {
	return _w1->_ar - _w2->_ar;
}

void MeteoModel::Interface::getStiff() {
	ge_d rst(_w1->_ar - _w2->_ar), arst(rst);
	if (arst < 0.0)
		arst *= -1.0;
	_p = rst / (arst + 1.0);
}

void MeteoModel::Interface::getWind() {
	ge_d aw(_w);
	if (_w < 0.0)
		_w *= -1.0;
	_w1->_w -= aw;
	_w2->_w -= aw;
	_w = WIND_TRANSFERT_RATE * _p * (_w1->_ma._t - _w2->_ma._t) * _w1->_wir * _w2->_wir;
	if (_w*_p > 0.0)
		_w = 0.0;
	else {
		if (_w > 0.0)
			aw = _w;
		else  aw = -1.0*_w;
		_w1->_w += aw;;
		_w2->_w += aw;;
	}
}

void MeteoModel::Interface::airMove() {
	ge_d dt, dw;
	if (_w != 0.0) {
		dt = _w * (_w1->_ma._t - _w2->_ma._t);
		dw = _w * (_w1->_ma._t - _w2->_ma._t);
		_w1->_ma._t += dt;
		_w2->_ma._t -= dt;
		_w1->_ma._w += dw;
		_w2->_ma._w -= dw;
	}
}

void MeteoModel::Interface::waterMove() {
	ge_d dw(_p * WATER_TRANSFERT_RATE * _w1->_war * _w2->_war);
	if (_p > 0.0 && _w1->_mg._w > 0.0) {
		dw *= _w1->_mg._w;
		_w1->_mg._w -= dw;
		_w2->_mg._w += dw;
		if (_w1->_s > 0.0) {
			ge_d ds(_p*SNOW_TRANSFERT_RATE*_w1->_s);
			_w1->_s -= ds;
			_w2->_s += ds;
		}
	}
	else if (_p < 0.0 && _w2->_mg._w > 0.0) {
		dw *= _w2->_mg._w;
		_w1->_mg._w -= dw;
		_w2->_mg._w += dw;
		if (_w2->_s > 0.0) {
			ge_d ds(_p*SNOW_TRANSFERT_RATE*_w2->_s);
			_w1->_s -= ds;
			_w2->_s += ds;
		}
	}
}
