#pragma once
#include "CudaFreqDrawer.h"
#include <SoundHandler.h>
#include "CudaAnalyticalVib.h"

struct pos_d {
	ge_d w, h;
};

__device__ void setPixel(void* pixel_surface, const int &x, const int &y, CudaSdlInterface::Parameter *param, const Uint32 &pixel);

__device__ Uint32 colorFromComplexRatio(const thrust::complex<ge_d> &c);

__device__ Uint32 colorFromComplex(const thrust::complex<ge_d> &c, const ge_d& ratio);

__device__ Uint32 colorFromMultiComplex(const thrust::complex<ge_d>& c1, const thrust::complex<ge_d>& c2, const ge_d& ratio);

__device__ Uint32 colorFromScalar(const ge_d& scalar, const ge_d& from, const ge_d& range);

__device__ Uint32 colorFromScalarBlackWhite(const ge_d& scalar, const ge_d& center, const ge_d& range);

__global__ void copy_complex_board(thrust::complex<ge_d> * c1, thrust::complex<ge_d>* c2, 
	void *pixel_surface, CudaSdlInterface::Parameter *param, CudaSdlInterface::NumberDrawParam* n_param);

__global__ void mandelbrot_set(thrust::complex<ge_d> * cur, thrust::complex<ge_d> * plan, CudaSdlInterface::Parameter *param);

__global__ void back_in_range(thrust::complex<ge_d> * cur, ge_pd *range, CudaSdlInterface::Parameter *param);

__global__ void equation_1(thrust::complex<ge_d>* cur, thrust::complex<ge_d>* plan, CudaSdlInterface::Parameter * param);

__global__ void apply_transformation(thrust::complex<ge_d>* cur, thrust::complex<ge_d>* transform, CudaSdlInterface::Parameter * param);

__global__ void check_param(CudaSdlInterface::Parameter* param);

__global__ void propagate_transformation(thrust::complex<ge_d>* transform, thrust::complex<ge_d>* temp, CudaSdlInterface::Parameter* param, CudaFractalModel::FractalData* data);

__global__ void copy_board(thrust::complex<ge_d>* dst, thrust::complex<ge_d>* src, CudaSdlInterface::Parameter* param);

__global__ void copy_double_board(ge_d* val, void* pixel_surface, CudaSdlInterface::Parameter* param, CudaSdlInterface::NumberDrawParam* n_param);

namespace KernelCallers { // KernelCallers
	CUDA_ERROR check_exec(const std::string &s);

	CUDA_ERROR copyComplexBoard(thrust::complex<ge_d> * c1, thrust::complex<ge_d>* c2, void *pixel_surface, 
		uint dimension, CudaSdlInterface::Parameter *param, CudaSdlInterface::NumberDrawParam *n_param);

	CUDA_ERROR mandelbrotSet(thrust::complex<ge_d> * cur, thrust::complex<ge_d> * plan, uint dimension, CudaSdlInterface::Parameter *param);

	CUDA_ERROR equation1(thrust::complex<ge_d> * cur, thrust::complex<ge_d> * plan, uint dimension, CudaSdlInterface::Parameter *param);

	CUDA_ERROR backInRange(thrust::complex<ge_d> * cur, uint dimension, CudaFractalModel::FractalData *data, CudaSdlInterface::Parameter *param);

	CUDA_ERROR applyTransformation(thrust::complex<ge_d> * cur, thrust::complex<ge_d> * transform, uint dimension, CudaSdlInterface::Parameter *param);

	CUDA_ERROR copyBoard(thrust::complex<ge_d>* dst, thrust::complex<ge_d>* src, uint dimension, CudaSdlInterface::Parameter* param);

	CUDA_ERROR propagateTransformation(thrust::complex<ge_d>* transform, thrust::complex<ge_d>* temp, uint dimension, CudaSdlInterface::Parameter* param, CudaFractalModel::FractalData* data);

	CUDA_ERROR checkParam(CudaSdlInterface::Parameter* param);

	CUDA_ERROR copyDoubleBoard(ge_d* val, void* pixel_surface, uint dimension, CudaSdlInterface::Parameter* param, CudaSdlInterface::NumberDrawParam* n_param);
}

__device__ pos_d get_relative_position(const int& x, CudaSdlInterface::Parameter* param);

__device__ void get_abs_position(const int& id, ge_i &x, ge_i &y, CudaSdlInterface::Parameter* param);

__global__ void draw_frequency(FreqPick *freq, thrust::complex<ge_d> * transform, thrust::complex<ge_d>* plan, FreqDrawerData* freq_data, CudaSdlInterface::Parameter *param);

__global__ void to_freq_array(Complex* lbuffer, Complex* rbuffer, struct FreqPick* freq);

__global__ void interpolate_freq(FreqPick* freq, FreqPick* int_freq, FreqPick* int_freq_last, CudaSdlInterface::Parameter* param, FreqDrawerData* freq_data);

__global__ void invert_int_freq(FreqPick* int_freq, FreqPick* int_freq_last);

namespace KernelFreq {
	CUDA_ERROR drawFrequency(FreqPick *int_freq, thrust::complex<ge_d> * transform, thrust::complex<ge_d>* plan, uint dimension,
		FreqDrawerData *freq_data, CudaSdlInterface::Parameter *param);

	CUDA_ERROR toFreqArray(Complex* lbuffer, Complex* rbuffer, struct FreqPick* freq, uint dimension);

	CUDA_ERROR interpolateFreq(FreqPick* freq, FreqPick* int_freq, FreqPick* int_freq_last, uint dimension, CudaSdlInterface::Parameter* param, FreqDrawerData* freq_data);
}

__global__ void vibration_model1_acceleration(ge_d* a, ge_d* h, CudaSdlInterface::Parameter* param, CudaPlaneVib::VibData* vdata);


__global__ void vibration_model1_position(ge_d* a, ge_d* v, ge_d* h, ge_d* avg, CudaSdlInterface::Parameter* param, CudaPlaneVib::VibData* vdata);
 
__global__ void ana_vibration_model1(thrust::complex<ge_d>* cur, ge_d* h, CudaSdlInterface::Parameter* param, CudaAnalyticalVib::AnaVibData* avdata);

namespace KernelVib {
	CUDA_ERROR vibrationModel1Acc(ge_d *a, ge_d *v, ge_d *h, ge_d *avg, uint dimension, CudaSdlInterface::Parameter* param, CudaPlaneVib::VibData *vdata);

	CUDA_ERROR vibrationModel1Pos(ge_d* a, ge_d* v, ge_d* h, ge_d* avg, uint dimension, CudaSdlInterface::Parameter* param, CudaPlaneVib::VibData* vdata);

	CUDA_ERROR anaVibrationModel1(thrust::complex<ge_d>* cur, ge_d* h, uint dimension, CudaSdlInterface::Parameter* param, CudaAnalyticalVib::AnaVibData* avdata);
}