#pragma once
#include <Clocks.h>
#include <string>
#include <RtAudio.h>
#include "SoundOperation.h"


class SoundHandler
{
public:
	SoundHandler();
	~SoundHandler();

	void setFile(const std::string& name, const stk::StkFloat& rate = 48000.0);
	void configure(const ge_i& sampling_size, const stk::StkFloat& frequency, const stk::StkFloat& frame_duration /* ms */);

	unsigned long testFileLengh();

	stk::StkFloat getFFTindex(const stk::StkFloat& frequency);

	bool noData() const;

	void printDevices();
	bool nextTick();

	AnalysedFArray* consumme();
	StereoChannel* consummeSample();

	void testApi();

	void getDevice();

	void enablePanComputation(const bool &enable = true);

private:


	bool tickLocked, panComputationEnabled;
	stk::FileLoop inFile;
	stk::StkFloat sample_rate, freq, frame_rate;
	unsigned long fileSampleLeft, nbSample, sampleSize, frameSize, nbChannel, current;
	std::vector<StereoSample> sample;

	std::fstream dbg_out;
};

