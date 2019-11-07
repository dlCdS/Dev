#pragma once

#include <SetColourWidget.h>

#define POLAR_CONV_FACTOR 0.9
#define TRANSPORT_CONV_FACTOR 0.8
#define TRANSPORT_THRESHOLD 0.05

class ModelData {

public:
	ModelData() {}
	~ModelData() {}
	static ge_i counter;
	ge_vd u, p, // speed, polar
		du, dp; // surrounding quantity and polar
	ge_d b, w,  // polar coef, quantity
		dw;     // quantity grad
	ColorS color;

	void setColor(const Color &c, const ge_d &quantity) {
		color = ColorS(c);
		w = quantity;
	}

	void initialize() {
		p.randomVector();
		u = { 0,0 };
		du = { 0,0 };
		dp = { 0,0 };
		b = 1.0;
		w = 0;
		dw = 0.0;
		color = { 0, 0, 0, 0 };
	}
	void update() {

		/*
		if(dw!=0.0) {
			w += dw / 4.0 * TRANSPORT_CONV_FACTOR;
			dw = 0.0;

			color.normalize();
			if (w > 1.0)
				color._s = 1.0;
			else color._s = w;
		}
		*/

		if (w < 1.0) {
			b = 1.0 - w;
			dp.normalize();
			dp = (dp - p);
			p += dp * POLAR_CONV_FACTOR;
			p.normalize();
		}
		else b = 0.0;

		dp = { 0, 0 };
	}
};


class Interface {
public:
	Interface(ModelData *ri, ModelData *le, const ge_vd &direction) : r(ri), l(le), dm(0.0), o(0.0), dir(direction) {}
	ge_d dm, o; // quantity grad, polarity ortho
	ge_vd dir;
	ModelData *l, *r; // left and right element
	void updateData() {
		l->dp += r->p * r->b;
		r->dp += l->p * l->b;

		/*
		ge_d lgo(abs(l->p * dir)), rgo(abs(dir * r->p));

		dm = (l->w - r->w)/2;
		ge_d absdm = abs(dm);
		if(absdm > TRANSPORT_THRESHOLD){
			r->dw += dm * lgo;
			l->dw -= dm * rgo;
			if (dm > 0) {
				r->color = r->color + (l->color*(absdm / r->w));
			}
			else {
				l->color = l->color + (r->color*(absdm / l->w));
			}
		}
		*/
	}
};

class WaterColorModel
{
public:
	WaterColorModel();
	~WaterColorModel();

	void setPolarityWidget(SetColourWidget *polw);
	void setSize(const ge_pi &_size, const ge_i &zoom);
	void generate();

	void cycle();
	void colorDrop(const int &radius, const ge_d &quantity);

protected:
	void updateData();
	void updateInterface();
	void drawPolariy();

	void destroyModel();
	void initWidget(SetColourWidget *widget);
	ge_pi _size;
	ge_i _zoom;
	ModelData **_data;
	std::vector<Interface> _interface;
	SetColourWidget *_pol;
};

class WatercolorModel
{
public:
	WatercolorModel();
	~WatercolorModel();
};

