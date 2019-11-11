#pragma once
#include "CudaPlaneVib.h"
#include <SoundOperation.h>

class CudaAnalyticalVib : public CudaPlaneVib
{
public:

	struct AnaVibData {
		ge_d time;
		ge_d period;
		ge_d speed;
	};


	CudaAnalyticalVib();
	virtual ~CudaAnalyticalVib();

	virtual CUDA_ERROR generate(const ge_d& stiffness, const ge_i& iteration);

	void testIntegration();

	void set(const ge_d &time=0.0, const ge_d & period =100, const ge_d& speed = 0.1);

protected:

	virtual void mainLoop();


	void copyAvdata();

	AnaVibData avdata, *d_avdata;
	thrust::complex<ge_d>* d_c;
};

