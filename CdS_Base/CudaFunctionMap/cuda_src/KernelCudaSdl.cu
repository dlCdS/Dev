#include "KernelCudaSdl.h"

__device__ void setPixel(void * pixel_surface, const int &x, const int &y, CudaSdlInterface::Parameter *param, const Uint32 &pixel)
{
	int bpp = param->BytesPerPixel;
	/* Here p is the address to the pixel we want to set */
	Uint8 *p = (Uint8 *)pixel_surface + y * param->pitch + x * bpp;

	switch (bpp) {
	case 1:
		*p = pixel;
		break;

	case 2:
		*(Uint16 *)p = pixel;
		break;

	case 3:
		if (SDL_BYTEORDER == SDL_BIG_ENDIAN) {
			p[0] = (pixel >> 16) & 0xff;
			p[1] = (pixel >> 8) & 0xff;
			p[2] = pixel & 0xff;
		}
		else {
			p[0] = pixel & 0xff;
			p[1] = (pixel >> 8) & 0xff;
			p[2] = (pixel >> 16) & 0xff;
		}
		break;

	case 4:
		*(Uint32 *)p = pixel;
		break;
	}
}

__device__ Uint32 colorFromComplexRatio(const thrust::complex<ge_d>& c)
{
	ge_d r(0.0), g(0.0), b(0.0);
	b = c.real() * c.real() + c.imag() * c.imag();
	if(b>0.01) {
		b = sqrt(b);
		r = (1.0 + c.real() / b) * 255.0 / 2.0;
		g = (1.0 + c.imag() / b) * 255.0 / 2.0;
		b = thrust::arg(c) * 255.0 / M_PI / 2.0;
	}
	else {
		r = g = b = 0.0;
	}
	
	return A_MASK + (Uint8(r) << 16) + (Uint8(g) << 8) + Uint8(b);
}

__device__ Uint32 colorFromComplex(const thrust::complex<ge_d>& c, const ge_d &ratio)
{
	ge_d r(0.0), g(0.0), b(0.0);
	b = c.real() * c.real() + c.imag() * c.imag();
	b = sqrt(b)*ratio;
	r = c.real()*ratio;
	g = c.imag()*ratio;
	// b = thrust::arg(c) * 255.0 / M_PI;

	return A_MASK + (Uint8(r) << 16) + (Uint8(g) << 8) + Uint8(b);
}

__device__ Uint32 colorFromMultiComplex(const thrust::complex<ge_d>& c1, const thrust::complex<ge_d>& c2, const ge_d& ratio)
{
	ge_d r(0.0), g(0.0), b(0.0);
	b = c2.real() * c2.real() + c2.imag() * c2.imag();
	b = sqrt(b) * ratio;
	r = c1.real() * ratio;
	g = c1.imag() * ratio;
	// b = thrust::arg(c) * 255.0 / M_PI;

	return A_MASK + (Uint8(r) << 16) + (Uint8(g) << 8) + Uint8(b);
}

__device__ Uint32 colorFromScalar(const ge_d& scalar, const ge_d& from, const ge_d& range)
{
	ge_d loc_scalar((scalar - from) / range);
	Uint8 r(0), g(0), b(0);
	if (loc_scalar < 0.0) {
		loc_scalar += int(loc_scalar - 1.0) * (-1.0);

	}
	else if (loc_scalar > 1.0) {
		loc_scalar -= int(loc_scalar);

	}

	if (loc_scalar <= 0.5) {
		b = (Uint8)((1.0 - 2.0 * loc_scalar) * 255.0);
		if (loc_scalar >= 0.25)
			g = (Uint8)((4.0 * loc_scalar - 1.0) * 255.0);
	}
	else {
		r = (Uint8)((2.0 * loc_scalar - 1.0) * 255.0);
		if (loc_scalar <= 0.75)
			g = (Uint8)((3.0 - 4.0 * loc_scalar) * 255.0);
	}
	return A_MASK + (r << 16) + (g << 8) + b;
}

__device__ Uint32 colorFromScalarBlackWhite(const ge_d& scalar, const ge_d& from, const ge_d& range)
{
	ge_d loc_scalar((scalar - from + range/2) / range*2.0);
	Uint8 w(0);
	if (loc_scalar < 0.0) loc_scalar *= -1.0;

	if (loc_scalar>1.0) {
		w = 0.0;
	}
	else {
		loc_scalar = 255.0 * (1.0 - loc_scalar);
		w = loc_scalar;
	}
	
	return A_MASK + (w << 16) + (w << 8) + w;
}

__global__ void copy_complex_board(thrust::complex<ge_d>* c1, thrust::complex<ge_d>* c2, 
	void * pixel_surface, CudaSdlInterface::Parameter * param, CudaSdlInterface::NumberDrawParam* n_param)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	if (id < param->size) {
		//Uint32 p = colorFromComplexRatio(c1[id]);
		Uint32 p = colorFromMultiComplex(c1[id], c2[id], n_param->complex_range);
		setPixel(pixel_surface, id%param->w, id / param->w, param,  p);
	}
}

__global__ void mandelbrot_set(thrust::complex<ge_d>* cur, thrust::complex<ge_d>* plan, CudaSdlInterface::Parameter * param)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	if (id < param->size) {
		cur[id] = cur[id] * cur[id] + plan[id];
	}
}

__global__ void back_in_range(thrust::complex<ge_d>* cur, ge_pd * range, CudaSdlInterface::Parameter * param)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	if (id < param->size)
	{
		ge_d val;
		if (cur[id].real() < -1 * range->w) {
			val = cur[id].real() + 2.0 * range->w * int(-1.0 * cur[id].real() / range->w / 2.0 + 2.0);
			cur[id].real(val);
		}
		else if (cur[id].real() >= range->w) {
			val = cur[id].real() - (int(cur[id].real() / range->w / 2.0) * 2.0 * range->w);
			cur[id].real(val);
		}

		if (cur[id].imag() < -1 * range->h) {
			val = cur[id].imag() + 2.0 * range->h * int(-1.0 * cur[id].imag() / range->h / 2.0 + 2.0);
			cur[id].imag(val);
		}
		else if (cur[id].imag() >= range->h) {
			val = cur[id].imag() - (int(cur[id].imag() / range->h / 2.0) * 2.0 * range->h);
			cur[id].imag(val);
		}
	}
}

__global__ void equation_1(thrust::complex<ge_d>* cur, thrust::complex<ge_d>* plan, CudaSdlInterface::Parameter * param)
{
	thrust::complex<ge_d> turn(0.10, 0.1), move(1.0, 0.0);;
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	if (id < param->size) {
		cur[id] = cur[id] * cur[id] + plan[id] ;
	}
}

__global__ void apply_transformation(thrust::complex<ge_d>* cur, thrust::complex<ge_d>* transform, CudaSdlInterface::Parameter * param)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	if (id < param->size) {
		cur[id] /= 100.0;
		cur[id] = transform[id];

		//cur[id] += transform[id];
		ge_d* c = (ge_d*)&transform[id];
		//c[0] = 0.0;
		//c[1] = 0.0;
	}
}

__global__ void check_param(CudaSdlInterface::Parameter* param)
{
	int loc = param->h;
}

__global__ void propagate_transformation(thrust::complex<ge_d>* transform, thrust::complex<ge_d>* temp, CudaSdlInterface::Parameter* param, CudaFractalModel::FractalData* data)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	int increment;
	if (id < param->size) {
		int x(id % param->w), y(id / param->w);
		int cur_pos(id);
		ge_d dist_center(param->w/2-x);
		dist_center /= param->w / 2;
		dist_center *= dist_center;
		dist_center *= sqrt(dist_center);

		if (data->transport != 0) {
			if (x > param->w / 2)
				x -= data->transport;
			else
				x += data->transport;
			if (x >= 0 && x < param->w)
				cur_pos = x + y * param->w;
		}

		ge_d dist, rad(data->prop_radius* data->prop_radius);
		ge_d divider(0.0), dir;
		//increment = 4.0*rad /num_calculus+1;
		increment = 1;
		temp[id] = 0.0;
		for(int i=-data->prop_radius; i< data->prop_radius; i+= increment)
			for (int j = -data->prop_radius; j < data->prop_radius; j+= increment) {

				dist = i * i + j * j;
				if (x + i >= 0 && x + i < param->w 
					&& y + j >= 0 && y + j < param->h 
					&& dist< rad){
					cur_pos = (x + i) + (y + j) * param->w;

					temp[id] += transform[cur_pos] * (1.0 - dist / rad) / rad * (data->prop_ratio + dist_center * data->prop_variation);
				}
			
			}
	}
}

__global__ void copy_board(thrust::complex<ge_d>* dst, thrust::complex<ge_d>* src, CudaSdlInterface::Parameter* param)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < param->size)
		dst[id] = src[id];

}

__global__ void copy_double_board(ge_d* val, void* pixel_surface, CudaSdlInterface::Parameter* param, CudaSdlInterface::NumberDrawParam* n_param)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < param->size) {
		Uint32 p = colorFromScalar(val[id], n_param->from, n_param->range);
		setPixel(pixel_surface, id % param->w, id / param->w, param, p);
	}
}

__device__ pos_d get_relative_position(const int& x, CudaSdlInterface::Parameter* param)
{
	pos_d p;
	p.w = x % param->w;
	p.h = x / param->w;
	p.w = p.w / (ge_d)param->w;
	p.h = p.h / (ge_d)param->h;
	return p;
}

void get_abs_position(const int& id, ge_i &x, ge_i& y, CudaSdlInterface::Parameter* param)
{
	x = id % param->w;
	y = id / param->w;
}

__global__ void draw_frequency(FreqPick *freq, thrust::complex<ge_d>* transform, thrust::complex<ge_d>* plan, FreqDrawerData* freq_data, CudaSdlInterface::Parameter * param)
{
	int id = threadIdx.x + blockIdx.x*blockDim.x;
	const bool proportionate(false);
	// get relative position in board
	pos_d p = get_relative_position(id, param);
	if (id < param->size) {
		
		ge_d ref(freq_data->average*2.0),
			divider(freq_data->max - ref);
		int i = param->h - id/ param->w -1;
		ge_d tmp, rtmp;
		if (i >= 0 && i < param->h) {
			
			// check tile is in pan range
			tmp = (freq[i].stereo + 1.0) / 2.0 - p.w;
			if (freq[i].stereo * freq[i].stereo > freq_data->radius) {
				tmp /= 8.0;
			}

			tmp *= tmp;
			rtmp = freq_data->radius * freq_data->radius;
			if (tmp <= rtmp) {
				// add the amplitude divided by the distance
				if (freq[i].amp >= ref) {
					if(proportionate)
						transform[id] += (freq[i].amp - ref) / divider * plan[id] * (1.0 - tmp / rtmp);
					else 
						transform[id] += plan[id] * (1.0 - tmp / rtmp);
				}
			}
		}
		
		
	}
}

__global__ void to_freq_array(Complex* lbuffer, Complex* rbuffer, FreqPick* freq)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	stk::StkFloat r, l;
	ge_d* cast;
	Complex c = rbuffer[i];
	cast = (ge_d*) &rbuffer[i];
	r = cast[0] * cast[0] + cast[1] *cast[1];
	cast = (ge_d*) &(lbuffer[i]);
	l = cast[0] * cast[0] + cast[1] * cast[1];
	//r = sqrt(r);
	//l = sqrt(l);
	freq[i].freq = i;
	freq[i].amp = r + l;
	freq[i].stereo = (r - l) / (r + l);
}

__global__ void interpolate_freq(FreqPick* freq, FreqPick* int_freq, FreqPick* int_freq_last, CudaSdlInterface::Parameter* param, FreqDrawerData* freq_data)
{
	// cell of the new interpolated must get a value
	// will be working on the last and then copied on the new
	int id = threadIdx.x + blockIdx.x * blockDim.x, src_id;
	const ge_d a = 1.7819935104 / 8192.0 , b = 5.26;
	ge_d ratio(ge_d(param->h) / ge_d(freq_data->freq_size)), d_src(ge_d(id)/ratio/ge_d(freq_data->freq_size)), delta;
	d_src = a * ge_d(freq_data->sample_size) * exp(b * d_src) - a * ge_d(freq_data->sample_size) * exp(b * freq_data->min_index/ ge_d(freq_data->freq_size));
	int prev(d_src), next(d_src + 1.0);
	delta = (d_src - ge_d(prev));
	FreqPick rel_freq;
	rel_freq.amp = freq[prev].amp + (freq[next].amp - freq[prev].amp) * delta;
	rel_freq.stereo = freq[prev].stereo + (freq[next].stereo - freq[prev].stereo) * delta; 


	src_id = a * exp(b * id);
	int_freq[id] = rel_freq;

	// id

}

__global__ void invert_int_freq(FreqPick* int_freq, FreqPick* int_freq_last)
{
	FreqPick* tmp = int_freq;
	int_freq = int_freq_last;
	int_freq_last = tmp;
}

__global__ void vibration_model1_acceleration(ge_d* a, ge_d* h, CudaSdlInterface::Parameter* param, CudaPlaneVib::VibData* vdata)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	const ge_d rev_sqrt2(0.7071067623), sqrt2(1.41421356237309);
	bool yup(false), ydown(false);
	if (id < param->size) {
		int x(id % param->w), y(id / param->w), cur_pos;
		
		for (int i = -1; i <= 1; i++) {
			if (x + i >= 0 && x + i < param->w) 
			for (int j = -1; j <= 1; j++) {
				if (y + j >= 0 && y + j < param->h) {
					if (i != 0 || j != 0) { // not the current element
						if (i * i + j * j <= 1) { // one of the direct neighboring tile
							// cur_pos = (x + i) + (y + j) * param->w;
							a[id] += h[(x + i) + (y + j) * param->w] - h[id];
						}
						else { // one of the corner tile
							// cur_pos = (x + i) + (y + j) * param->w;
							a[id] += (h[(x + i) + (y + j) * param->w] - h[id])/ sqrt2;
						}
					}
				}
			}
		}
		
		/*
		if (y - 1 >= 0) ydown = true;
		if (y + 1 < param->h) yup = true;

		if (x - 1 >= 0) {
			a[id] += h[(x - 1) + (y) * param->w] - h[id];
			if(yup)
				a[id] += (h[(x - 1) + (y + 1)*param->w] - h[id]) * rev_sqrt2;
			if (ydown)
				a[id] += (h[(x - 1) + (y - 1) * param->w] - h[id]) * rev_sqrt2;
		}
		if (x + 1 < param->w) {
			a[id] += h[(x + 1) + (y)*param->w] - h[id];
			if (yup)
				a[id] += (h[(x + 1) + (y + 1) * param->w] - h[id]) * rev_sqrt2;
			if (ydown)
				a[id] += (h[(x + 1) + (y - 1) * param->w] - h[id]) * rev_sqrt2;
		} 
		if (ydown) a[id] += h[(x)+(y - 1) * param->w] - h[id];
		if (yup) a[id] += h[(x)+(y + 1) * param->w] - h[id];
		*/
		a[id] /= (8.0 + 4.0 * sqrt2);
	}
}

__global__ void vibration_model1_acceleration(char* a, char* h, CudaSdlInterface::Parameter* param, CudaPlaneVib::VibData* vdata)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	const ge_d rev_sqrt2(0.7071067623);
	bool yup(false), ydown(false);
	if (id < param->size) {
		int x(id % param->w), y(id / param->w), cur_pos;
		if (y - 1 >= 0) ydown = true;
		if (y + 1 < param->h) yup = true;

		if (x - 1 >= 0) {
			a[id] += h[(x - 1) + (y)*param->w] - h[id];
			if (yup)
				a[id] += (h[(x - 1) + (y + 1) * param->w] - h[id]) * rev_sqrt2;
			if (ydown)
				a[id] += (h[(x - 1) + (y - 1) * param->w] - h[id]) * rev_sqrt2;
		}
		if (x + 1 < param->w) {
			a[id] += h[(x + 1) + (y)*param->w] - h[id];
			if (yup)
				a[id] += (h[(x + 1) + (y + 1) * param->w] - h[id]) * rev_sqrt2;
			if (ydown)
				a[id] += (h[(x + 1) + (y - 1) * param->w] - h[id]) * rev_sqrt2;
		}
		if (ydown) a[id] += h[(x)+(y - 1) * param->w] - h[id];
		if (yup) a[id] += h[(x)+(y + 1) * param->w] - h[id];

		a[id] *= 0.07322330470336; // (8.0 + 4.0 * sqrt2);
	}
}

__global__ void vibration_model1_position(ge_d* a, ge_d* v, ge_d* h, ge_d* avg, CudaSdlInterface::Parameter* param, CudaPlaneVib::VibData* vdata)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < param->size) {
		v[id] += a[id] * vdata->stiffness;
		h[id] += v[id];
		a[id] = 0.0;
	}
}


__global__ void vibration_model1_position(char* a, char* v, char* h, char* avg, CudaSdlInterface::Parameter* param, CudaPlaneVib::VibData* vdata)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < param->size) {
		v[id] += a[id] * vdata->stiffness;
		h[id] += v[id];
		a[id] = 0.0;
	}
}

__global__ void ana_vibration_model1(thrust::complex<ge_d> * cur, ge_d* h, CudaSdlInterface::Parameter* param, CudaAnalyticalVib::AnaVibData* avdata)
{
	int id = threadIdx.x + blockIdx.x * blockDim.x;
	if (id < param->size) { 
		ge_i x, y, range(3);
		get_abs_position(id, x, y, param);
		((ge_d*)(&cur[id]))[0] = 0.0;
		((ge_d*)(&cur[id]))[1] = 0.0;

		for (int j = -range; j <= range; j++) {
			for (int i = -range; i <= range; i++) {
				ge_d phi = (x + ge_d(j * param->w) - param->w / 2) * (x + ge_d(j * param->w) - param->w / 2) + (y + ge_d(i * param->h) - param->h / 2) * (y + ge_d(i * param->h) - param->h / 2);
				phi = sqrt(phi);
				cur[id] += thrust::polar(1.0, -2 * M_PI / avdata->period * avdata->time + phi / avdata->speed);
			}

		}

		h[id] = ((ge_d*)(&cur[id]))[0];
	}
}

CUDA_ERROR KernelCallers::check_exec(const std::string & s)
{
	cudaError_t cudaStatus = cudaGetLastError();
	if (cudaStatus != CUDA_SUCCESS) {
		Log(LERROR, "kernel function ", s, " failed : ", cudaGetErrorString(cudaStatus));
		return cudaStatus;
	}
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != CUDA_SUCCESS) {
		Log(LERROR, "function ", s, " failed to synchronize ", cudaStatus);
	}
	return cudaStatus;
}

CUDA_ERROR KernelCallers::copyComplexBoard(thrust::complex<ge_d>* c1, thrust::complex<ge_d>* c2, void * pixel_surface, 
	uint dimension, CudaSdlInterface::Parameter * param, CudaSdlInterface::NumberDrawParam* n_param)
{
	copy_complex_board << < (dimension / THREADS_PER_BLOCK +1), THREADS_PER_BLOCK, 1 >>>(c1, c2, pixel_surface, param, n_param);
	return check_exec("copyComplexBoard");
}

CUDA_ERROR KernelCallers::mandelbrotSet(thrust::complex<ge_d>* cur, thrust::complex<ge_d>* plan, uint dimension, CudaSdlInterface::Parameter * param)
{
	mandelbrot_set << < (dimension / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK, 1 >> > (cur, plan, param);
	return check_exec("mandelbrotSet");
}

CUDA_ERROR KernelCallers::equation1(thrust::complex<ge_d>* cur, thrust::complex<ge_d>* plan, uint dimension, CudaSdlInterface::Parameter * param)
{
	equation_1 << < (dimension / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK, 1 >> > (cur, plan, param);
	return check_exec("equation_1");
}

CUDA_ERROR KernelCallers::backInRange(thrust::complex<ge_d>* cur, uint dimension, CudaFractalModel::FractalData* data, CudaSdlInterface::Parameter * param)
{
	back_in_range << < (dimension / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK, 1 >> > (cur, &data->tore, param);
	return check_exec("backInRange");
}

CUDA_ERROR KernelCallers::applyTransformation(thrust::complex<ge_d>* cur, thrust::complex<ge_d>* transform, uint dimension, CudaSdlInterface::Parameter * param)
{
	apply_transformation << < (dimension / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK, 1 >> > (cur, transform, param);
	return check_exec("applyTransformation");
}

CUDA_ERROR KernelCallers::copyBoard(thrust::complex<ge_d>* dst, thrust::complex<ge_d>* src, uint dimension, CudaSdlInterface::Parameter* param)
{
	copy_board << < (dimension / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK, 1 >> > (dst, src, param);
	return check_exec("copyBoard");
}

CUDA_ERROR KernelCallers::propagateTransformation(thrust::complex<ge_d>* transform, thrust::complex<ge_d>* temp, uint dimension, CudaSdlInterface::Parameter* param, CudaFractalModel::FractalData* data)
{
	
	propagate_transformation << < (dimension / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK, 1 >> > (transform, temp, param, data);
	CUDA_ERROR status = check_exec("propagateTransformation");
	if (status != CUDA_SUCCESS)
		return status;
	copy_board << < (dimension / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK, 1 >> > (transform, temp, param);
	return check_exec("copy_board");
}

CUDA_ERROR KernelCallers::checkParam(CudaSdlInterface::Parameter* param)
{
	check_param << < 1, 1, 1 >> > (param);
	return check_exec("checkParam");
}

CUDA_ERROR KernelCallers::copyDoubleBoard(ge_d* val, void* pixel_surface, uint dimension, CudaSdlInterface::Parameter* param, CudaSdlInterface::NumberDrawParam* n_param)
{
	copy_double_board << < (dimension / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK, 1 >> > (val, pixel_surface, param, n_param);
	return check_exec("copyCompcopy_double_boardlexBoard");
}

CUDA_ERROR KernelFreq::drawFrequency(FreqPick* freq, thrust::complex<ge_d>* transform, thrust::complex<ge_d>* plan, uint dimension, FreqDrawerData* freq_data, CudaSdlInterface::Parameter* param)
{
	draw_frequency << < (dimension / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK, 1 >> > (freq, transform, plan, freq_data, param);
	return KernelCallers::check_exec("drawFrequency");
}

CUDA_ERROR KernelFreq::toFreqArray(Complex* lbuffer, Complex* rbuffer, FreqPick* freq, uint dimension)
{
	to_freq_array << < (dimension / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK, 1 >> > (lbuffer, rbuffer, freq);
	return KernelCallers::check_exec("toFreqArray");
}

CUDA_ERROR KernelFreq::interpolateFreq(FreqPick* freq, FreqPick* int_freq, FreqPick* int_freq_last, uint dimension, CudaSdlInterface::Parameter* param, FreqDrawerData* freq_data)
{
	interpolate_freq << < (dimension / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK, 1 >> > (freq, int_freq, int_freq_last, param, freq_data);
	return KernelCallers::check_exec("interpolateFreq");
	invert_int_freq << < 1, 1, 1 >> > (int_freq, int_freq_last);
	return KernelCallers::check_exec("invert_int_freq");
}

CUDA_ERROR KernelVib::vibrationModel1Acc(ge_d* a, ge_d* v, ge_d* h, ge_d* avg, uint dimension, CudaSdlInterface::Parameter* param, CudaPlaneVib::VibData* vdata)
{
	vibration_model1_acceleration << < (dimension / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK, 1 >> > (a, h, param, vdata);
	return KernelCallers::check_exec("vibration_model1_acceleration");
}

CUDA_ERROR KernelVib::vibrationModel1Pos(ge_d* a, ge_d* v, ge_d* h, ge_d* avg, uint dimension, CudaSdlInterface::Parameter* param, CudaPlaneVib::VibData* vdata)
{

	vibration_model1_position << < (dimension / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK, 1 >> > (a, v, h, avg, param, vdata);
	return KernelCallers::check_exec("vibration_model1_position");
}

CUDA_ERROR KernelVib::vibrationModel1Acc(char* a, char* v, char* h, char* avg, uint dimension, CudaSdlInterface::Parameter* param, CudaPlaneVib::VibData* vdata)
{
	vibration_model1_acceleration << < (dimension / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK, 1 >> > (a, h, param, vdata);
	return KernelCallers::check_exec("vibration_model1_acceleration");
}

CUDA_ERROR KernelVib::vibrationModel1Pos(char* a, char* v, char* h, char* avg, uint dimension, CudaSdlInterface::Parameter* param, CudaPlaneVib::VibData* vdata)
{

	vibration_model1_position << < (dimension / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK, 1 >> > (a, v, h, avg, param, vdata);
	return KernelCallers::check_exec("vibration_model1_position");
}

CUDA_ERROR KernelVib::anaVibrationModel1(thrust::complex<ge_d> * cur, ge_d* h, uint dimension, CudaSdlInterface::Parameter* param, CudaAnalyticalVib::AnaVibData* avdata)
{
	ana_vibration_model1 << < (dimension / THREADS_PER_BLOCK + 1), THREADS_PER_BLOCK, 1 >> > (cur, h, param, avdata);
	CUDA_ERROR status = KernelCallers::check_exec("ana_vibration_model1");
	if (status != CUDA_SUCCESS)
		return status;

	return status;
}
