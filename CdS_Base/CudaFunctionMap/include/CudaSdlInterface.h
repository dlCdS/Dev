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

	CudaSdlInterface();
	virtual ~CudaSdlInterface();

	virtual void setColourWidget(SetColourWidget* scw);


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

	struct Parameter sdl_param, * d_sdl_param;
};

