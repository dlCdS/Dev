#pragma once
#include "CudaSdlInterface.h"

class CudaPlaneVib :
	public CudaSdlInterface
{
	struct VibData {
		ge_d stiffness;
	};

	CudaPlaneVib();
	virtual ~CudaPlaneVib();

	virtual CUDA_ERROR generate(const ge_d& stiffness);


protected:

	VibData vdata, * d_vdata;
	ge_d* d_a,
		* d_v,
		* d_h,
		* d_avg;
};

