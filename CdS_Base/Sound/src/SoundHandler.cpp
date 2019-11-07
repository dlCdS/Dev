#include "SoundHandler.h"

SoundHandler::SoundHandler() : panComputationEnabled(true)
{
	dbg_out.open("rl_ticks.csv", std::ios::out | std::ios::trunc);
}

SoundHandler::~SoundHandler()
{
}

void SoundHandler::setFile(const std::string& name, const stk::StkFloat& rate)
{
	stk::Stk::setSampleRate(rate);
	inFile.openFile(name.c_str(), false);
	nbChannel = inFile.channelsOut();
	sample_rate = rate;
	Log(LINFO, "Open file ", name, ", ", nbChannel, " channels, ", sample_rate, "hz");
}

void SoundHandler::configure(const ge_i& sampling_size, const stk::StkFloat& frequency, const stk::StkFloat& frame_duration)
{
	fileSampleLeft = inFile.getSize();
	stk::StkFloat adapt_ratio;
	adapt_ratio= (fileSampleLeft / sample_rate);
	// adapt_ratio = 1.0;
	// inFile.setFrequency(frequency/ adapt_ratio);
	sampleSize = sampling_size;
	freq = sample_rate / frequency;
	frameSize = freq * frame_duration / 1000;
	nbSample = sampleSize / frameSize + 1;


	sample.resize(nbSample);
	for (int i = 0; i < nbSample; i++)
		sample[i].setSize(sampleSize, frameSize, i, nbChannel, freq);

	current = 0;
	Log(LINFO, "Configured ", sampleSize, " sample per buffer, ",
		frameSize, " sample per frame (", sample_rate, "hz/", frequency, "*", frame_duration, "/1000), ",
		nbSample, " samples in parallel (", frameSize * nbSample, ") ",
		fileSampleLeft, " total samples");
	testFileLengh();
}

unsigned long SoundHandler::testFileLengh()
{
	const int last_size(100);
	stk::StkFloat last[last_size], first_sig, cur_sig(0.0);
	unsigned long lengh(0);
	for(int i=0; i<last_size;i++){
		last[i] = inFile.tick(0);
		// Log(LINFO, "Tick ", i, " is ", last[i]);
	}

	first_sig = 0.0;
	for (int i = 0; i < last_size; i++)
		first_sig += last[i];

	lengh = last_size;
	while (first_sig != cur_sig && lengh < fileSampleLeft + 1000) {
		last[lengh% last_size] = inFile.tick(0);

		cur_sig = 0.0;
		for (int i = 0; i < last_size; i++)
			cur_sig += last[i];

		lengh++;

	}
	inFile.reset();
	Log(LINFO, "Signature matched at ", lengh, " for ", fileSampleLeft, " expected");
	// fileSampleLeft = lengh;
	return lengh;
}

stk::StkFloat SoundHandler::getFFTindex(const stk::StkFloat& frequency) {
	return ge_d(sampleSize) * frequency / freq;
}

bool SoundHandler::noData() const
{
	return fileSampleLeft <= 0;
}


void SoundHandler::printDevices()
{
	// RtAudio *audio = new RtAudio();
	/*
	int device = audio->getDeviceCount();
	RtAudio::DeviceInfo info;
	for (int i = 1; i <= device; i++) {
		info = audio->getDeviceInfo(i);
		Log(LINFO, "device ", i, " ", info.name);
	}
	*/
}

bool SoundHandler::nextTick()
{
	static ge_i count(0);
	stk::StkFloat t[2];
	if (!tickLocked) {

		// Compute filled samples
		for (auto s = sample.begin(); s < sample.end(); ++s) {
			if (s->filled) {
				if (s->computed) {
					//Log(LINFO, count, " fileSampleLeft : ----- computed");
					tickLocked = true;
				}
				else {
					s->compute(panComputationEnabled);
					//Log(LINFO, count, " fileSampleLeft : -- do compute");
				}
			}
			else {
				// Log(LINFO, count, " fileSampleLeft : not filled");
			}
		}



		if (!tickLocked) {
			count++;
			// get ticks
			for (int j = 0; j < nbChannel; j++) {
				t[j] = inFile.lastOut(j);
			}
			inFile.tick();

			for (auto s = sample.begin(); s < sample.end(); ++s) {
				s->addTick(t[0], t[1]);
			}

			if (fileSampleLeft != 0)
				fileSampleLeft--;
			else
				tickLocked = true;
		}
	}
	return !tickLocked;
}

AnalysedFArray* SoundHandler::consumme()
{
	static int count(0);
	// Log(LINFO, "Consumed sample ", count++);
	if (sample[current].computed) {
		
		stk::StkFloat lmax(0.0);
		ge_i id;
		for (int i = 0; i < sample[current].freq.freq.size(); i++) {
			if (sample[current].freq.freq[i].amp > lmax) {
				lmax = sample[current].freq.freq[i].amp;
				id = i;
			}
		}
		// dbg_out << id << std::endl;
		
		AnalysedFArray* freq = &sample[current].freq;
		sample[current].computed = false;
		current = (current + 1) % nbSample;
		tickLocked = false;
		return freq;
	}
	else {
		return NULL;
	}
}

StereoChannel * SoundHandler::consummeSample()
{
	if (sample[current].computed) {
		StereoChannel* channel = &sample[current].channel;
		sample[current].computed = false;
		current = (current + 1) % nbSample;
		tickLocked = false;
		return channel;
	}
	else {
		return NULL;
	}
}

void SoundHandler::testApi()
{
	FArray* freq;
	AnalysedFArray* ana_freq;
	stk::StkFloat max, val;
	ge_i id, count(0);
	while (!noData() && count < 1000) {
		// Log(LINFO, "Reading file ", fileSampleLeft);
		Clocks::start("FreqLoop");
		Clocks::start("gatherTicks");
		while (nextTick()) {
			;;
		}
		Clocks::stop("gatherTicks");
		// Log(LINFO, "Consumming");
		Clocks::start("consumme");
		ana_freq = consumme();
		freq = &ana_freq->freq;
		max = 0.0;
		id = 0;
		for (int i = 0; i < freq->size(); i++) {
			val = freq[0][i].amp;
			if (freq[0][i].amp > max) {
				max = freq[0][i].amp;
				id = i;
			}
		}
		Log(LINFO, "TestApi : sample ", count++, "\tmax freq : ", freq[0][id].freq, "hz \t", freq[0][id].amp, "\tstereo : ", freq[0][id].stereo);
		for (int i = 0; i < freq->size(); i++)
			freq[0][i] = { 0.0, 0.0, 0.0 };
		Clocks::stop("consumme");
		Clocks::stop("FreqLoop");
	}
}

void SoundHandler::getDevice()
{/*

	MMDeviceEnumerator pEnumerator;
	const CLSID CLSID_MMDeviceEnumerator = __uuidof(MMDeviceEnumerator);
	const IID IID_IMMDeviceEnumerator = __uuidof(IMMDeviceEnumerator);
	HRESULT hr = CoCreateInstance(
		CLSID_MMDeviceEnumerator, NULL,
		CLSCTX_ALL, IID_IMMDeviceEnumerator,
		(void**)&pEnumerator);

		*/
}

void SoundHandler::enablePanComputation(const bool & enable)
{
	panComputationEnabled = enable;
}


FreqSpectrum::FreqSpectrum() : threshold(500000.0)
{
}

FreqSpectrum::~FreqSpectrum()
{
}

void FreqSpectrum::fromRaw(const CArray& array, const stk::StkFloat& sample_rate)
{
}

void FreqSpectrum::fromRaw(const CArray& right, const CArray& left, const stk::StkFloat& sample_rate)
{
	CArray buffer;
	stk::StkFloat r, l;
	FreqPick freq;
	for (int i = 0; i < right.size() / 2; i++) {
		r = right[i].imag() * right[i].imag() + right[i].real() * right[i].real();
		l = left[i].imag() * left[i].imag() + left[i].real() * left[i].real();
		if (r + l > threshold) {
			r = sqrt(r);
			l = sqrt(l);
			freq.freq = i * sample_rate / (stk::StkFloat)(right.size());
			freq.amp = r + l;
			freq.stereo = (r - l) / (r + l);
			spec.push_back(freq);
			freq.print();
		}
	}
}

/* void FFT::fft(CArray& x)
{
	const size_t N = x.size();
	if (N <= 1) return;

	// divide
	CArray even = x[std::slice(0, N / 2, 2)];
	CArray  odd = x[std::slice(1, N / 2, 2)];

	// conquer
	fft(even);
	fft(odd);

	// combine
	for (size_t k = 0; k < N / 2; ++k)
	{
		Complex t = std::polar(1.0, -2 * STK_PI * k / N) * odd[k];
		x[k] = even[k] + t;
		x[k + N / 2] = even[k] - t;
	}
}

*/

void FFT::optimized_fft(CArray& x)
{
	// DFT
	unsigned int N = x.size(), k = N, n;
	double thetaT = 3.14159265358979323846264338328L / N;
	// thetaT /= 2.0;
	Complex phiT = Complex(cos(thetaT), -sin(thetaT)), T;
	while (k > 1)
	{
		n = k;
		k >>= 1;
		phiT *= phiT;
		T = 1.0L;
		for (unsigned int l = 0; l < k; l++)
		{
			for (unsigned int a = l; a < N; a += n)
			{
				unsigned int b = a + k;
				Complex t = x[a] - x[b];
				x[a] += x[b];
				x[b] = t * T;
			}
			T *= phiT;
		}
	}
	// Decimate
	unsigned int m = (unsigned int)log2(N);
	for (unsigned int a = 0; a < N; a++)
	{
		unsigned int b = a;
		// Reverse bits
		b = (((b & 0xaaaaaaaa) >> 1) | ((b & 0x55555555) << 1));
		b = (((b & 0xcccccccc) >> 2) | ((b & 0x33333333) << 2));
		b = (((b & 0xf0f0f0f0) >> 4) | ((b & 0x0f0f0f0f) << 4));
		b = (((b & 0xff00ff00) >> 8) | ((b & 0x00ff00ff) << 8));
		b = ((b >> 16) | (b << 16)) >> (32 - m);
		if (b > a)
		{
			Complex t = x[a];
			x[a] = x[b];
			x[b] = t;
		}
	}
	//// Normalize (This section make it not working correctly)
	//Complex f = 1.0 / sqrt(N);
	//for (unsigned int i = 0; i < N; i++)
	//	x[i] *= f;
}

stk::StkFloat FFT::getNorm(const Complex& c)
{
	return sqrt(c.imag() * c.imag() + c.real() * c.real());
}

void StereoSample::setSize(const ge_i& bufferSize, const ge_i& frameSize, const ge_i& id, const ge_i& nbchannel, const stk::StkFloat& sample_freq)
{
	sampleFreq = sample_freq;
	freq.freq.resize(bufferSize / 2);
	nbChannel = nbchannel;
	size = bufferSize;
	currentTick = 0 - id * frameSize;
	for (int i = 0; i < 2; i++)
		channel.buffer[i].resize(bufferSize);
	filled = false;
	computed = false;
	sampleOffset = (bufferSize / frameSize + 1) * frameSize - bufferSize;
	Log(LINFO, "Created StereoSample ", sampleOffset, " offset currently at ", currentTick, " frame ", frameSize, " buffer ", bufferSize);
}

void StereoSample::addTick(const stk::StkFloat& ltick, const stk::StkFloat& rtick)
{
	if (currentTick >= 0) {
		channel.buffer[0][currentTick] = { ltick, 0.0 };
		channel.buffer[1][currentTick] = { rtick, 0.0 };
		
	}
	currentTick++;
	if (currentTick >= size) {
		filled = true;
		currentTick = -sampleOffset;
	}
	
}

void StereoSample::compute(const bool &enabled)
{
	if(enabled){
		for (int j = 0; j < nbChannel; j++)
			FFT::optimized_fft(channel.buffer[j]);
		if (nbChannel == 1)
			toFreqArray(channel.buffer[0]);
		else if (nbChannel == 2)
			toFreqArray(&channel.buffer[0], &channel.buffer[1]);
		stk::StkFloat val, lmax(0);
		ge_i id(0);
		for (int i = 0; i < channel.buffer[0].size(); i++) {
			val = channel.buffer[0][i].real();
			if (val > lmax) {
				lmax = val;
				id = i;
			}
		}
		// Log(LINFO, "TestApi : \t", id, " id \t", val, "\tstereo : ");
	}

	
	computed = true;
	filled = false;
}

void StereoSample::toFreqArray(const CArray& array)
{
}

void StereoSample::toFreqArray(const CArray *right, const CArray *left)
{
	stk::StkFloat r, l;
	freq.min=-1.0;
	freq.max = 0.0;
	freq.average = 0.0;
	ge_i id(0);
	for (int i = 0; i < right->size() / 2; i++) {
		stk::StkFloat *cast,  * lcast, *rcast;
		cast = (stk::StkFloat*) &(*right)[i];
		r = cast[0] * cast[0] + cast[1] * cast[1];
		cast = (stk::StkFloat*) &(*left)[i];
		l = cast[0] * cast[0] + cast[1] * cast[1];

		// r = (*right)[i].imag() * (*right)[i].imag() + (*right)[i].real() * (*right)[i].real();
		// l = (*left)[i].imag() * (*left)[i].imag() + (*left)[i].real() * (*left)[i].real();

		r = sqrt(r);
		l = sqrt(l);
		r = sqrt(r);
		l = sqrt(l);
		freq.freq[i].freq = i * sampleFreq / (stk::StkFloat)((*right).size());
		freq.freq[i].amp = r + l;
		freq.freq[i].stereo = (l - r) / (r + l);

		freq.average+= freq.freq[i].amp;
		if(freq.min<0.0) freq.min = freq.freq[i].amp;
		else if(freq.min> freq.freq[i].amp)freq.min = freq.freq[i].amp;

		if (freq.freq[i].amp > freq.max) {
			freq.max = freq.freq[i].amp;
			id = i;
			// Log(LINFO, "Raw snd  ", i, " : ", freq.freq[i].amp, " value ", freq.max);
		}
	}
	freq.average /= right->size() / 2;
	// Log(LINFO, "To freq array max : ", id, " / ", right->size() / 2, " : ", freq.freq[id].freq, "hz ", freq.freq[id].amp, " min ", freq.min, " max ", freq.max, " avg ", freq.average, " stereo ", freq.freq[id].stereo);
}
