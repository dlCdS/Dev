#pragma once

#include <VirtualModel.h>
#include <thrust/complex.h>
#include "CudaConfiguration.h"


CUDA_ERROR cudaMallocErrorHandle(void** dst, const size_t &size, const std::string &name);

class CudaSdlInterface : public VirtualModel
{
public:
	struct Parameter {
		int BytesPerPixel;
		int h, w, size;
		int pitch;
	};

	struct NumberDrawParam {
		ge_d from, range; // param for scalar drawing
		ge_d complex_range; // param for complex number
	};

	CudaSdlInterface();
	virtual ~CudaSdlInterface();

	virtual void setColourWidget(SetColourWidget* scw);
	CUDA_ERROR setNumberDrawParam(const NumberDrawParam& param = { -0.1, 0.2, 255.0 });

	CUDA_ERROR testMemcpy();

protected:

	CUDA_ERROR preCheck();
	CUDA_ERROR setSurface(SDL_Surface *surface);

	CUDA_ERROR copySurfaceToDevice();
	CUDA_ERROR copySurfaceToHost();

	CUDA_ERROR initCuda();

	size_t surfaceSize();

	SDL_Surface * _surface;
	void *d_surface_pixel;

	Parameter sdl_param, * d_sdl_param;
	NumberDrawParam * d_num_param;
};

