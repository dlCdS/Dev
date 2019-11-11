#include "CudaAnalyticalVib.h"
#include "KernelCudaSdl.h"

CudaAnalyticalVib::CudaAnalyticalVib()
{
}

CudaAnalyticalVib::~CudaAnalyticalVib()
{
}

CUDA_ERROR CudaAnalyticalVib::generate(const ge_d& stiffness, const ge_i& iteration)
{
	CUDA_ERROR status = CudaPlaneVib::generate(stiffness, iteration);

	cudaMallocErrorHandle((void**)&d_avdata, sizeof(AnaVibData), "d_avdata");

	thrust::complex<ge_d>* c = new thrust::complex<ge_d>[_size.w * _size.h];
	size_t size = _size.w * _size.h * sizeof(std::complex<ge_d>);

	status = cudaMallocErrorHandle((void**)&d_c, size, "d_c");
	if (status != CUDA_SUCCESS)
		return (CUDA_ERROR)-1;

	for (int i = 0; i < _size.w; i++)
		for (int j = 0; j < _size.h; j++)
			c[i + _size.w * j] = thrust::complex<ge_d>(0.0, 0.0);

	status = cudaMemcpy(d_c, c, size, cudaMemcpyHostToDevice);
	if (status != CUDA_SUCCESS) {
		Log(LERROR, "failed to copy d_c ", cudaGetErrorString(status));
		return status;
	}

}

void CudaAnalyticalVib::testIntegration()
{
	std::fstream f("test_integral.csv", std::ios::out | std::ios::trunc);
	Complex c, res({ 0.0, 0.0 });
	ge_i range = 1000;
	ge_d omegv = 1.0, x = 0.5, xp = 0.2, y = 0.5, yp = 0.1, w = 1.0, h = 1.0,
		phi, step = 0.1, speed;
	for (speed =0.0; speed <50.0; speed += step) {
		res = { 0.0, 0.0 };
		for (int j = -range; j <= range; j++) {
			for (int i = -range; i <= range; i++) {
				phi = (x + ge_d(j * h) - xp) * (x + ge_d(j * h) - xp) + (y + ge_d(i * w) - yp) * (y + ge_d(i * w) - yp); 
				phi = sqrt(phi);
				res += thrust::polar(1.0, -2 * M_PI * omegv + phi / speed);
			}
			
		}
		std::cout << speed << std::endl;
		f << std::fixed << speed << ";" << res.real() << ";" << res.imag() << std::endl;
	}

	f.close();
}

void CudaAnalyticalVib::set(const ge_d& time, const ge_d& period, const ge_d& speed)
{
	avdata.time = time;
	avdata.period = period;
	avdata.speed = speed;

	copyAvdata();
	
}

void CudaAnalyticalVib::mainLoop()
{
	KernelVib::anaVibrationModel1(d_c, d_h, _size.w * _size.h, d_sdl_param, d_avdata);
}

void CudaAnalyticalVib::copyAvdata()
{
	CUDA_ERROR status = cudaMemcpy(d_avdata, &avdata, sizeof(AnaVibData), cudaMemcpyHostToDevice);
	if (status != CUDA_SUCCESS)
		Log(LERROR, "Failed to copy avdata");
}
