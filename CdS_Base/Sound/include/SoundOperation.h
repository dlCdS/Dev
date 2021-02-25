#pragma once
#include <Common.h>
#include <FileLoop.h>
#include <complex>
#include <iostream>
#include <vector>
#include <fstream>

const double STK_PI = 3.141592653589793238460;

typedef std::complex<ge_d> Complex;
typedef std::vector<Complex> CArray;



// freq = index * sample_rate / sample_size / 8

namespace FFT {
	// void fft(CArray& x);
	void optimized_fft(CArray& x);
	stk::StkFloat getNorm(const Complex& c);
}

struct FreqPick {
	stk::StkFloat freq, amp, stereo;
	void print() {
		std::cout << freq << " hz " << amp << " : pan " << stereo << std::endl;
	}
};

typedef std::vector<FreqPick> FArray;

class FreqSpectrum {
public:
	FreqSpectrum();
	~FreqSpectrum();

	void fromRaw(const CArray& array, const stk::StkFloat& sample_rate);
	void fromRaw(const CArray& right, const CArray& left, const stk::StkFloat& sample_rate);


private:
	std::vector<FreqPick> spec;
	stk::StkFloat threshold;
};

struct StereoChannel {
	CArray buffer[2];
};

struct AnalysedFArray {
	FArray freq;
	stk::StkFloat average, max, min;
};

struct StereoSample {
	StereoChannel channel;
	AnalysedFArray freq;
	int currentTick, size, sampleOffset, nbChannel;
	bool filled, computed;
	stk::StkFloat sampleFreq;

	void setSize(const ge_i& bufferSize, const ge_i& frameSize, const ge_i& id, const ge_i& nbChannel, const stk::StkFloat& sample_rate);
	void addTick(const stk::StkFloat& ltick, const stk::StkFloat& rtick);
	void compute(const bool &enabled=true);
	void toFreqArray(const CArray& array);
	void toFreqArray(const CArray * right, const CArray * left);
};
