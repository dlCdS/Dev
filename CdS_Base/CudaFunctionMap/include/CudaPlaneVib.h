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

	ge_i addSource(const ge_pd& rel_pos);
	void setPosition(const ge_d& value, const ge_i& src_id);

protected:

	struct PosToPtr {
		ge_pd pos;
		ge_i offset;
	};

	virtual void draw();

	virtual void mainLoop();

	std::vector<PosToPtr> _sources;
	VibData vdata, * d_vdata;
	ge_d* d_a,
		* d_v,
		* d_h,
		* d_avg;
};

