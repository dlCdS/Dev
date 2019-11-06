#pragma once
#include "CudaFractalModel.h"
#include <SoundHandler.h>

struct DeviceStereoChannel {
	Complex *lbuffer;
	Complex *rbuffer;
	struct FreqPick *freq;
};

struct FreqDrawerData {
	uint freq_size, sample_size;
	ge_d radius, average, min, max, min_index;
	FreqDrawerData() : freq_size(0), radius(0.001) {}
};

class CudaFreqDrawer : public CudaFractalModel
{
public:
	

	CudaFreqDrawer();
	~CudaFreqDrawer();

	void setFrequencyIndexInterval(const ge_d& min_index, const ge_d& max_index, const int &sample_size);

	int consume(StereoChannel* stereo_channel);
	int consume(AnalysedFArray* freq_pick);

	void process();

	void draw();

	void testComputedFreq();

protected:

	void setup(const int &size, const int& sample_size);

	void freqInterpolation();

	struct DeviceStereoChannel *d_channel, channel;
	struct FreqDrawerData* d_freq_data, freq_data;
	FreqPick *d_int_freq, *d_int_freq_last;

};

