#include "CudaAnalyticalVib.h"

CudaAnalyticalVib::CudaAnalyticalVib()
{
}

CudaAnalyticalVib::~CudaAnalyticalVib()
{
}

void CudaAnalyticalVib::testIntegration()
{
	std::fstream f("test_integral.csv", std::ios::out | std::ios::trunc);
	Complex c, res({ 0.0, 0.0 });
	ge_i range = 1000;
	ge_d omegv = 1.0, x = 0.5, xp = 0.5, y = 0.0, yp = 0.1, w = 1.0,
		phi, step=0.01;
	for (xp =-4.0*w; xp < 5.0*w; xp += step) {
		res = { 0.0, 0.0 };
		for (int i = -range; i <= range; i++) {
			phi = (x - xp) * (x - xp) + (y + i * w - yp) * (y + i * w - yp);
			res += thrust::polar(1.0, 2 * M_PI * omegv * phi);
		}
		f << xp << ";" << res.real() << std::endl;
	}

	f.close();
}
