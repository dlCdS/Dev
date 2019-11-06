#pragma once
#include <string>
#include <SDL.h>
#include "XMLCompliant.h"
#include <Clocks.h>

/* MATH */




/* SDL */

#define DEFAULT_WINDOW_SIZE ge_pi(1200, 700)

#define DEFAULT_R 150
#define DEFAULT_G 150
#define DEFAULT_B 150

#define MASK 0xff00ff00

const uint32_t R_MASK = 0x00ff0000;
const uint32_t G_MASK = 0x0000ff00;
const uint32_t B_MASK = 0x000000ff;
const uint32_t A_MASK = 0xff000000;

struct Color {
	Color() : _r(DEFAULT_R), _g(DEFAULT_G), _b(DEFAULT_B) {}
	Color(const Uint8 &r, const Uint8 &g, const Uint8 &b) : _r(r), _g(g), _b(b) {}
	Color(const Uint32 &c) : _r((c & R_MASK) >> 16), _g((c & G_MASK) >> 8), _b(c & B_MASK) {}
	Uint32 get() const { return A_MASK + (_r << 16) + (_g << 8) + _b; }
	Uint8 _r, _g, _b;
	void setColor(SDL_Renderer *renderer, const Uint8 & alpha = 255) const;

	static Color ColorFromScalar(const ge_d& scalar, const ge_d& from, const ge_d& range) {
		ge_d loc_scalar((scalar - from)/range);
		if (loc_scalar < 0.0){
			loc_scalar += int(loc_scalar - 1.0) * (-1.0);
			// Log(LINFO, " scalar conv ", loc_scalar, " + ", int(loc_scalar - 1.0) * (-1.0), " = ", loc_scalar + ge_d(mod));

		}
		else if (loc_scalar > 1.0) {
			loc_scalar -= int(loc_scalar);
			// Log(LINFO, " scalar conv ", loc_scalar, " + ", int(loc_scalar), " - ", loc_scalar - ge_d(mod));

		}
		Color color(0, 0, 0);

		if (loc_scalar <= 0.5){
			color._b = (Uint8)((1.0 - 2.0 * loc_scalar) * 255.0);
			if (loc_scalar >= 0.25)
				color._g = (Uint8)((4.0 * loc_scalar - 1.0) * 255.0);
		}
		else {
			color._r = (Uint8)((2.0 * loc_scalar - 1.0) * 255.0);
			if (loc_scalar <= 0.75)
				color._g = (Uint8)((3.0 - 4.0 * loc_scalar) * 255.0);
		}
		return color;
	}
};

struct ColorS {
	ColorS(const Color &c=Color()) {
		normalize(c);
	}

	ColorS(const ge_d &r, const ge_d &g, const ge_d &b, const ge_d &s)
		: _r(r), _g(g), _b(b), _s(s) {}

	void normalize(const Color &c) {
		Uint8 sup = max(c._r, c._g);
		sup = max(sup, c._b);
		if (sup > 0) {
			_r = (ge_d)c._r / sup;
			_g = (ge_d)c._g / sup;
			_b = (ge_d)c._b / sup;
			_s = (ge_d)sup / 255;
		}
		else {
			_r = _g = _b = _s = 0.0;
		}
	}

	void normalize() {
		ge_d sup = max(_r, _g);
		sup = max(sup, _b);
		if (sup > 0) {
			_r /= sup;
			_g /= sup;
			_b /= sup;
			_s = 1.0;
		}
		else {
			_r = _g = _b = _s = 0.0;
		}
	}

	Color getColor() const {
		return { (Uint8)((ge_d) 255 * _r*_s), (Uint8)((ge_d)255 * _g*_s), (Uint8)((ge_d)255 * _b*_s )};
	}

	ColorS operator*(const ge_d &d) {
		return { _r*d, _g*d, _b*d, _s };
	}
	ColorS operator+(const ColorS &c) {
		ColorS tc(_r + c._r, _g + c._g, _b + c._b, 0.0);
		tc.normalize();
		return tc;
	}
	
	ge_d _r, _g, _b, _s;
};