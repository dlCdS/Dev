  #pragma once
#include "CudaSdlInterface.h"


class CudaFractalModel : public CudaSdlInterface
{
public:
	CudaFractalModel();
	~CudaFractalModel();

	struct FractalData {
		ge_pd tore;
		ge_d relaxation;
		ge_d prop_radius;
		ge_d prop_ratio, prop_variation;
		ge_i transport;
	};

	virtual CUDA_ERROR generate(const ge_pd& center, const ge_d& degree_angle, const ge_pd& contraction, const ge_pd& tore = { 1.0, 1.0 }, const ge_d &prop_radius=5.0, const ge_d &relaxation=1.0);

protected:

	virtual void draw();

	virtual void mainLoop();

	FractalData fdata, * d_fdata;
	thrust::complex<ge_d>* d_c,
		* d_cur,
		* d_transform,
		* d_temp;
};

