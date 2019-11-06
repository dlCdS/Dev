#include "CudaSdlInterface.h"
#include "KernelCudaSdl.h"

CudaSdlInterface::CudaSdlInterface() : d_surface_pixel(NULL)
{

}


CudaSdlInterface::~CudaSdlInterface()
{
}

void CudaSdlInterface::setColourWidget(SetColourWidget * scw)
{
	VirtualModel::setColourWidget(scw);
	setSurface(_scw->getSurface());
}

CUDA_ERROR CudaSdlInterface::setSurface(SDL_Surface * surface)
{

	Log(LINFO, "setting surface");

	_surface = surface;


	sdl_param.w = _surface->w;
	sdl_param.h = _surface->h;
	sdl_param.size = sdl_param.w* sdl_param.h;

	sdl_param.BytesPerPixel = _surface->format->BytesPerPixel;
	sdl_param.pitch = _surface->pitch;


	Log(LINFO, "surface setted ", surface, " ", sdl_param.w, " ", sdl_param.h);

	cudaError_t cudaStatus = preCheck();
	if (cudaStatus != CUDA_SUCCESS) {
		Log(LERROR, "failed to init cuda");
		return cudaStatus;
	}

	cudaStatus = initCuda();
	if (cudaStatus != CUDA_SUCCESS) {
		Log(LERROR, "failed to init device data");
		return cudaStatus;
	}

	return CUDA_SUCCESS;
}

CUDA_ERROR CudaSdlInterface::testMemcpy()
{
	cudaError_t cudaStatus;

	cudaStatus = copySurfaceToDevice();
	if (cudaStatus != CUDA_SUCCESS) {
		Log(LERROR, "failed tocopySurfaceToDevice");
		return cudaStatus;
	}

	for(int i=0; i< sdl_param.w; i++)
		for (int j = 0; j < sdl_param.h; j++) {
			Surface::putpixel(_surface, i, j, 0);
		}

	cudaStatus = copySurfaceToHost();
	if (cudaStatus != CUDA_SUCCESS) {
		Log(LERROR, "failed copySurfaceToHost");
		return cudaStatus;
	}

	return CUDA_SUCCESS;
}


CUDA_ERROR CudaSdlInterface::initCuda()
{
	cudaError_t cudaStatus;


	Log(LINFO, "allocating surface of size ", surfaceSize());

	cudaStatus = cudaMallocErrorHandle((void**)&d_surface_pixel, surfaceSize(), "d_surface_pixel");
	if (cudaStatus != CUDA_SUCCESS)
		return (CUDA_ERROR)-1;

	cudaStatus = cudaMallocErrorHandle((void**)&d_sdl_param, sizeof(CudaSdlInterface::Parameter), "parameter");
	if (cudaStatus != CUDA_SUCCESS){
		return (CUDA_ERROR)-1;
	}

	Log(LINFO, "Size of CudaSdlInterface::Parameter is ", sizeof(CudaSdlInterface::Parameter));
	//sdl_param.pitch = 700;
	cudaStatus = cudaMemcpy(d_sdl_param, &sdl_param, sizeof(CudaSdlInterface::Parameter), cudaMemcpyHostToDevice);
	if (cudaStatus != CUDA_SUCCESS) {
		Log(LERROR, "Failed to copy sdl_param");
		return (CUDA_ERROR)-1;
	}

	KernelCallers::checkParam(d_sdl_param);

	return CUDA_SUCCESS;
}

CUDA_ERROR CudaSdlInterface::preCheck()
{
	cudaError_t cudaStatus;
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != CUDA_SUCCESS) {
		Log(LERROR, "failed to set device");
		return cudaStatus;
	}
	Log(LINFO, "device setted");
	return CUDA_SUCCESS;
}

CUDA_ERROR CudaSdlInterface::copySurfaceToDevice()
{
	SDL_LockSurface(_surface);
	cudaError_t cudaStatus = cudaMemcpy(d_surface_pixel, _surface->pixels, surfaceSize(), cudaMemcpyHostToDevice);
	SDL_UnlockSurface(_surface);
	return cudaStatus;
}

CUDA_ERROR CudaSdlInterface::copySurfaceToHost()
{
	return cudaMemcpy(_surface->pixels, d_surface_pixel, surfaceSize(), cudaMemcpyDeviceToHost);
}

size_t CudaSdlInterface::surfaceSize()
{
	return _surface->h * _surface->pitch;
}

CUDA_ERROR cudaMallocErrorHandle(void ** dst, const size_t & size, const std::string & name)
{
	CUDA_ERROR cudaStatus = cudaMalloc(dst, size);
	if (cudaStatus != CUDA_SUCCESS) {
		Log(LERROR, "failed to allocate " + name, ", ", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	return CUDA_SUCCESS;
}
