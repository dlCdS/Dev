#pragma once
#include "CudaSdlInterface.h"

class CudaPlaneVib :
	public CudaSdlInterface
{
	CudaPlaneVib();
	virtual ~CudaPlaneVib();

protected:

	ge_d * d_a,
		* d_v,
		* d_h;
};

