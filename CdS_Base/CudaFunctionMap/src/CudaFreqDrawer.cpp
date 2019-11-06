#include "CudaFreqDrawer.h"
#include "KernelCudaSdl.h"


CudaFreqDrawer::CudaFreqDrawer() : CudaFractalModel(), d_channel(NULL), d_freq_data(NULL)
{
}


CudaFreqDrawer::~CudaFreqDrawer()
{
}

void CudaFreqDrawer::setFrequencyIndexInterval(const ge_d& min_index, const ge_d& max_index, const int& sample_size)
{
	freq_data.freq_size = max_index- min_index;
	freq_data.min_index = min_index;
	freq_data.sample_size = sample_size;
	setup(max_index, sample_size);
}

int CudaFreqDrawer::consume(StereoChannel*  stereo_channel)
{
	if (d_channel == NULL) {
		setup(stereo_channel->buffer[0].size()/2, stereo_channel->buffer[0].size());
	}

	CUDA_ERROR status = cudaMemcpy(channel.lbuffer, &(stereo_channel->buffer[0][0]), freq_data.freq_size  * sizeof(Complex), cudaMemcpyHostToDevice);
	if (status != CUDA_SUCCESS) {
		Log(LERROR, "failed copy lbuffer");
	}
	status = cudaMemcpy(channel.rbuffer, &stereo_channel->buffer[1][0], freq_data.freq_size * sizeof(Complex), cudaMemcpyHostToDevice);
	if (status != CUDA_SUCCESS) {
		Log(LERROR, "failed copy rbuffer");
	}


	/* DEBUG
	ge_d bmax(0.0);
	ge_i id(0);
	for (int i = 0; i < freq_data.freq_size; i++) {
		if (FFT::getNorm(stereo_channel->buffer[0][i]) > bmax) {
			bmax = FFT::getNorm(stereo_channel->buffer[0][i]);
			id = i;
		}
		if (i == 0)
			Log(LINFO, "Raw stch  ", i, " : ", stereo_channel->buffer[0][i]);
	}
	Log(LINFO, "Test src buffer max is : ", id, " ", bmax, " copied to ", channel.lbuffer);
	
	Complex* lbuffer = new Complex[freq_data.freq_size];
	status = cudaMemcpy(lbuffer, channel.lbuffer, freq_data.freq_size * sizeof(Complex), cudaMemcpyDeviceToHost);
	if (status != CUDA_SUCCESS)
		Log(LERROR, "Failed to get computed lbuffer");
	for (int i = 0; i < freq_data.freq_size; i++) {
		if (FFT::getNorm(lbuffer[i]) > bmax) {
			bmax = FFT::getNorm(lbuffer[i]);
			id = i;
			if (lbuffer[i] != stereo_channel->buffer[0][i])
				Log(LINFO, "Buffer diff ", i, lbuffer[i], " ", stereo_channel->buffer[0][i]);
		}
	}
	Log(LINFO, "Test dst buffer max is : ", id, " ", bmax, " copied from ", channel.lbuffer);

	delete[] lbuffer;
	*/
	
	return 0;
}

int CudaFreqDrawer::consume(AnalysedFArray* freq_pick)
{
	if (d_channel == NULL) {
		setup(freq_pick->freq.size()*2, freq_pick->freq.size() * 2);
	}

	freq_data.average = freq_pick->average;
	freq_data.min = freq_pick->min;
	freq_data.max = freq_pick->max;



	CUDA_ERROR status = cudaMemcpy(channel.freq, &((freq_pick->freq)[0]), freq_data.freq_size * sizeof(FreqPick), cudaMemcpyHostToDevice);
	if (status != CUDA_SUCCESS) {
		Log(LERROR, "failed copy freq pick");
	}

	status = cudaMemcpy(d_freq_data, &freq_data, sizeof(FreqDrawerData), cudaMemcpyHostToDevice);
	if (status != CUDA_SUCCESS) {
		Log(LERROR, "failed copy fdata");
	}

	KernelFreq::interpolateFreq(channel.freq, d_int_freq, d_int_freq_last, sdl_param.h, d_sdl_param, d_freq_data);

	/*  DEBUG
	struct FreqPick* freq = new FreqPick[freq_data.freq_size];
	CUDA_ERROR status = cudaMemcpy(freq, channel.freq, freq_data.freq_size * sizeof(FreqPick), cudaMemcpyDeviceToHost);
	if (status != CUDA_SUCCESS)
		Log(LERROR, "Failed to get computed freq");

	ge_d bmax(0.0);
	ge_i id(0);
	for (int i = 0; i < freq_data.freq_size; i++) {
		if (freq[i].amp > bmax) {
			bmax = freq[i].amp;
			id = i;
			Log(LINFO, "Raw cuda ", i, " : ", freq[i].amp);
		}
	}
	Log(LINFO, "TestComputedFreq max is : ", id, " ", freq[id].amp, " stereo ", freq[id].stereo, " left ", lbuffer[id], " right ", rbuffer[id]);
	*/
	return 0;
}

void CudaFreqDrawer::process()
{
	// KernelFreq::toFreqArray(channel.lbuffer, channel.rbuffer, channel.freq, freq_data.freq_size);
	// testComputedFreq();
	KernelFreq::drawFrequency(d_int_freq, d_transform, d_c, _size.w * _size.h, d_freq_data, d_sdl_param);
	KernelCallers::propagateTransformation(d_transform, d_temp, _size.w * _size.h, d_sdl_param, d_fdata);
	KernelCallers::backInRange(d_transform, _size.w * _size.h, d_fdata, d_sdl_param);
	KernelCallers::applyTransformation(d_cur, d_transform, _size.w * _size.h, d_sdl_param);
	//KernelCallers::equation1(d_cur, d_c, _size.w * _size.h, d_sdl_param);
	KernelCallers::backInRange(d_cur, _size.w * _size.h, d_fdata, d_sdl_param);
}

void CudaFreqDrawer::draw()
{
	CudaFractalModel::draw();
}

void CudaFreqDrawer::testComputedFreq()
{
	struct FreqPick* freq = new FreqPick[freq_data.freq_size];
	Complex* lbuffer = new Complex[freq_data.freq_size];
	Complex* rbuffer = new Complex[freq_data.freq_size];

	CUDA_ERROR status = cudaMemcpy(freq, channel.freq, freq_data.freq_size * sizeof(FreqPick), cudaMemcpyDeviceToHost);
	if (status != CUDA_SUCCESS)
		Log(LERROR, "Failed to get computed freq");

	status = cudaMemcpy(lbuffer, channel.lbuffer, freq_data.freq_size * sizeof(Complex), cudaMemcpyDeviceToHost);
	if (status != CUDA_SUCCESS)
		Log(LERROR, "Failed to get computed lbuffer");

	status = cudaMemcpy(rbuffer, channel.rbuffer, freq_data.freq_size * sizeof(Complex), cudaMemcpyDeviceToHost);
	if (status != CUDA_SUCCESS)
		Log(LERROR, "Failed to get computed rbuffer");

	else {
		ge_d bmax(0.0);
		ge_i id(0);
		for (int i = 0; i < freq_data.freq_size; i++) {
			if (freq[i].amp > bmax) {
				bmax = freq[i].amp;
				id = i;
				Log(LINFO, "Raw cuda ", i, " : ", freq[i].amp);
			}
		}
		Log(LINFO, "TestComputedFreq max is : ", id, " ", freq[id].amp, " stereo ", freq[id].stereo, " left ", lbuffer[id], " right ", rbuffer[id]);

	}
	delete[] freq, lbuffer, rbuffer;
}

void CudaFreqDrawer::setup(const int & size, const int& sample_size)
{
	freq_data.freq_size = size;
	freq_data.sample_size = sample_size;
	Log(LINFO, "Setting drawer freq_size to ", freq_data.freq_size);
	CUDA_ERROR status = cudaMallocErrorHandle((void**)&d_channel, sizeof(DeviceStereoChannel), "DeviceStereoChannel");
	status = cudaMallocErrorHandle((void**)&channel.lbuffer, freq_data.freq_size * sizeof(Complex), "DeviceStereoChannel buffer 1");
	status = cudaMallocErrorHandle((void**)&channel.rbuffer, freq_data.freq_size * sizeof(Complex), "DeviceStereoChannel buffer 2");
	status = cudaMallocErrorHandle((void**)&channel.freq, freq_data.freq_size * sizeof(FreqPick), "DeviceStereoChannel freq");

	status = cudaMallocErrorHandle((void**)&d_freq_data, sizeof(FreqDrawerData), "frequency data");

	
	status = cudaMemcpy(d_freq_data, &freq_data, sizeof(FreqDrawerData), cudaMemcpyHostToDevice);
	if (status != CUDA_SUCCESS) {
		Log(LERROR, "failed copy fdata");
	}
	status = cudaMemcpy(d_channel, &channel, sizeof(DeviceStereoChannel), cudaMemcpyHostToDevice);
	if (status != CUDA_SUCCESS) {
		Log(LERROR, "failed copy d_channel");
	}

	// Allocating interpolated freq
	status = cudaMallocErrorHandle((void**)&d_int_freq, sdl_param.h * sizeof(FreqPick), "int_freq");
	status = cudaMallocErrorHandle((void**)&d_int_freq_last, sdl_param.h * sizeof(FreqPick), "int_freq_last");
}

void CudaFreqDrawer::freqInterpolation()
{
	// this function ineerpolate the freq array to a heigh fitting freq array on device
}
