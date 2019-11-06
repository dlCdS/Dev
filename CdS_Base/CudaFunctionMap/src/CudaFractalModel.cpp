#include "CudaFractalModel.h"
#include "KernelCudaSdl.h"



CudaFractalModel::CudaFractalModel() : CudaSdlInterface(), fdata({ {0.0,0.0}, 1.0, 1.0, 0.641, 0.025, 4})
{
}


CudaFractalModel::~CudaFractalModel()
{
}

CUDA_ERROR CudaFractalModel::generate(const ge_pd & center, const ge_d &degree_angle, const ge_pd & contraction, const ge_pd & tore, const ge_d& prop_radius, const ge_d& relaxation)
{
	CUDA_ERROR cudaStatus;
	thrust::complex<ge_d> *c = new thrust::complex<ge_d>[_size.w * _size.h];
	size_t size = _size.w * _size.h * sizeof(std::complex<ge_d>);

	Log(LINFO, "allocated complex board size ", size);

	cudaStatus = cudaMallocErrorHandle((void**)&d_c, size , "d_c");
	if (cudaStatus != CUDA_SUCCESS)
		return (CUDA_ERROR)-1;

	cudaStatus = cudaMallocErrorHandle((void**)&d_cur, size, "d_cur");
	if (cudaStatus != CUDA_SUCCESS)
		return (CUDA_ERROR)-1;

	cudaStatus = cudaMallocErrorHandle((void**)&d_transform, size, "d_transform");
	if (cudaStatus != CUDA_SUCCESS)
		return (CUDA_ERROR)-1;

	cudaStatus = cudaMallocErrorHandle((void**)&d_temp, size, "d_temp");
	if (cudaStatus != CUDA_SUCCESS)
		return (CUDA_ERROR)-1;  

	ge_d norm, phi(degree_angle * 2.0 * M_PI / 360.0);
	thrust::complex<ge_d> theta(cos(phi), sin(phi));
	const bool normalize(false);
	for (int i = 0; i < _size.w; i++) 
		for (int j = 0; j < _size.h; j++) {
			c[i + _size.w * j] = thrust::complex<ge_d>(ge_d(i - _size.w / 2) * 2.0 * contraction.w / ge_d(_size.w) + center.w,
				ge_d(j - _size.h / 2) * 2.0 * contraction.h / ge_d(_size.h) + center.h )* theta;
			
			if(normalize){
				norm = FFT::getNorm(c[i + _size.w * j]);
				if (norm > 0.000001)
					c[i + _size.w * j] *= norm;
			}
		}
	
	cudaStatus = cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);
	if (cudaStatus != CUDA_SUCCESS) {
		Log(LERROR, "failed to copy d_c ", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	

	for (int i = 0; i < _size.w; i++)
		for (int j = 0; j < _size.h; j++)
			c[i + _size.w * j] = thrust::complex<ge_d>( 0.0, 0.0 );

	cudaStatus = cudaMemcpy(d_transform, c, size, cudaMemcpyHostToDevice);
	if (cudaStatus != CUDA_SUCCESS) {
		Log(LERROR, "failed to copy d_transform ", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(d_temp, c, size, cudaMemcpyHostToDevice);
	if (cudaStatus != CUDA_SUCCESS) {
		Log(LERROR, "failed to copy d_temp ", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	cudaStatus = cudaMemcpy(d_cur, c, size, cudaMemcpyHostToDevice);
	if (cudaStatus != CUDA_SUCCESS) {
		Log(LERROR, "failed to copy d_cur ", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}

	fdata.tore = tore;
	fdata.relaxation = relaxation;
	fdata.prop_radius = prop_radius;

	cudaStatus = cudaMallocErrorHandle((void**)&d_fdata, sizeof(FractalData), " d_fdata");
	if (cudaStatus != CUDA_SUCCESS)
		return (CUDA_ERROR) -1;

	cudaStatus = cudaMemcpy(d_fdata, &fdata, sizeof(FractalData), cudaMemcpyHostToDevice);
	if (cudaStatus != CUDA_SUCCESS) {
		Log(LERROR, "failed to copy d_tore ", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
}

void CudaFractalModel::draw()
{
	CUDA_ERROR cudaStatus;
	KernelCallers::copyComplexBoard(d_cur, d_transform, d_surface_pixel, _size.w*_size.h, d_sdl_param);
	cudaStatus = copySurfaceToHost();
	if (cudaStatus != CUDA_SUCCESS) {
		Log(LERROR, "failed copySurfaceToHost");
	}
	_scw->updateTexture();

}

void CudaFractalModel::mainLoop()
{
	KernelCallers::equation1(d_cur, d_c, _size.w*_size.h, d_sdl_param);
	KernelCallers::backInRange(d_cur, _size.w*_size.h, d_fdata, d_sdl_param);
}
