#include "CudaPlaneVib.h"

CudaPlaneVib::CudaPlaneVib()
{
}

CudaPlaneVib::~CudaPlaneVib()
{
}

CUDA_ERROR CudaPlaneVib::generate(const ge_d& stiffness)
{
	CUDA_ERROR cudaStatus;
	ge_d* c = new ge_d[_size.w * _size.h];
	size_t size = _size.w * _size.h * sizeof(ge_d);

	Log(LINFO, "allocated vib board size ", size);

	cudaStatus = cudaMallocErrorHandle((void**)&d_a, size, "d_a");
	if (cudaStatus != CUDA_SUCCESS)
		return (CUDA_ERROR)-1;

	cudaStatus = cudaMallocErrorHandle((void**)&d_v, size, "d_v");
	if (cudaStatus != CUDA_SUCCESS)
		return (CUDA_ERROR)-1;

	cudaStatus = cudaMallocErrorHandle((void**)&d_h, size, "d_h");
	if (cudaStatus != CUDA_SUCCESS)
		return (CUDA_ERROR)-1;

	cudaStatus = cudaMallocErrorHandle((void**)&d_avg, size, "d_avg");
	if (cudaStatus != CUDA_SUCCESS)
		return (CUDA_ERROR)-1;

	for (int i = 0; i < _size.w; i++)
		for (int j = 0; j < _size.h; j++)
			c[i + _size.w * j] = 0.0;

	cudaStatus = cudaMemcpy(d_a, c, size, cudaMemcpyHostToDevice);
	if (cudaStatus != CUDA_SUCCESS) {
		Log(LERROR, "failed to copy d_c ", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(d_v, c, size, cudaMemcpyHostToDevice);
	if (cudaStatus != CUDA_SUCCESS) {
		Log(LERROR, "failed to copy d_transform ", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(d_h, c, size, cudaMemcpyHostToDevice);
	if (cudaStatus != CUDA_SUCCESS) {
		Log(LERROR, "failed to copy d_temp ", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(d_avg, c, size, cudaMemcpyHostToDevice);
	if (cudaStatus != CUDA_SUCCESS) {
		Log(LERROR, "failed to copy d_cur ", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	vdata.stiffness = stiffness;

	cudaStatus = cudaMallocErrorHandle((void**)&d_vdata, sizeof(VibData), " d_vdata");
	if (cudaStatus != CUDA_SUCCESS)
		return (CUDA_ERROR)-1;

	cudaStatus = cudaMemcpy(d_vdata, &vdata, sizeof(VibData), cudaMemcpyHostToDevice);
	if (cudaStatus != CUDA_SUCCESS) {
		Log(LERROR, "failed to d_vdata d_tore ", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
}
