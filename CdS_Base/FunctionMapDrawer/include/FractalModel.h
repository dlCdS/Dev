#pragma once

#include "VirtualModel.h"
#include <complex>

class FractalModel : public VirtualModel
{
public:
	FractalModel();
	~FractalModel();


	void generate(const ge_pd& center, const ge_pd& contraction, const ge_pd& tore = { 1.0, 1.0 });
	Color getRatioNormColor(const ge_i& i, const ge_i& j, std::complex<ge_d>** source);
	
	void publish();

protected:

	virtual void draw();

	virtual void mainLoop();


	ge_pd _center, _contraction, _tore;

	std::complex<ge_d> ** _c,
		** _cur,
		** _transform;

	void toreSpace(const ge_i& i, const ge_i& j, std::complex<ge_d>** source);
};

