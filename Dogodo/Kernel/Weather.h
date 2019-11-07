#pragma once
#include <XMLParsable.h>
#include <Clocks.h>

#define WIND_TRANSFERT_RATE 0.1
#define WATER_TRANSFERT_RATE 0.1
#define SNOW_TRANSFERT_RATE 0.02
#define HEAT_TRANSFERT_RATE 0.1
#define MOISTURE_TRANSFERT_RATE 0.2
#define RAIN_TRANSFERT_RATE 0.1
#define ALTITUDE_HEAT_LOSS 0.2
#define SEA_ALTITUDE_OFFSET 10.0
#define SNOW_WATER_TRANSFORM_RATE 0.2
#define SNOW_WATER_TRANSFORM_TRIGGER 0.01
#define ALMOST_ZERO 0.001

class Weather :
	public XML::Parsable
{
	struct Meteo {
		Meteo() : _t(0.0), _w(0.0) {}
		ge_d _t; // temperature
		ge_d _w; // water
	};
public:
	Weather();
	virtual ~Weather();

	Meteo _mg, _ma; // ground & air
	ge_d _s; // snow
	ge_d _w; // wind
	ge_d _m; // moisture
	ge_d _wir, _war; // wind and water movement reduction
	ge_d _a, _ar; // altitude, real altitude (+ water)
	bool _rain, _sea;

	void heatTransfert();
	void waterTransfert();
	void sunHeat(const ge_d &light);
	void altitudeCooling();
	void loop(const ge_d &light);

	virtual void associate();
	std::string XMLName() const;

	static ge_d _am; // max altitude
};

class MeteoModel :
	public XML::Parsable {
public:
	struct Interface {
		Interface(Weather *w1, Weather *w2) : _w1(w1), _w2(w2), _p(0.0), _w(0.0) {}
		Weather *_w1, *_w2;
		ge_d _p; // stiffness
		ge_d _w; // wind
		ge_d realStiff();
		void getStiff();
		void getWind();
		void airMove();
		void waterMove();
	};
public:
	virtual ~MeteoModel();

	static void setSize(const ge_pi &size);
	static void generate();
	static void test(const ge_pi &size, const int &ntest);

protected:
	std::string XMLName() const;
	virtual void associate();

	void step();
	void weatherLoop();
	void interfaceLoop();

	Weather **_m; // map
	std::vector<Interface> _i;
	ge_pi _s; // size
	ge_d _light;
	int _tempo, _offset;

	bool _locked;
	MeteoModel();
	static MeteoModel _sing;
};

