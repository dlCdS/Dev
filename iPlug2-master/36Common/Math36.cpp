#include "Math36.h"

Math36::AllpassFilter::AllpassFilter() : _x1(0.0), _x2(0.0), _y1(0.0), _y2(0.0)
{

}

double Math36::AllpassFilter::process(const double& x)
{
  double tmp = 2.0 * _pole.real() * _y1 - _pnorm * _pnorm * _y2 + _x2 - 2.0 * _pole.real() * _x1 + _pnorm * _pnorm * x;
  _y2 = _y1;
  _y1 = tmp;
  _x2 = _x1;
  _x1 = x;
  return tmp;
}

void Math36::Filter::setPole(const Complex& pole)
{
  _pole = pole;
  _pnorm = std::norm(_pole);
}

void Math36::fft(CArray& x, const int& N)
{
  // DFT
  unsigned int k = N, n;
  double thetaT = M36_PI / N;
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

void Math36::ifft(CArray& x, const int& N)
{
  // conjugate the complex numbers
  x = x.apply(std::conj);

  // forward fft
  fft(x, N);

  // conjugate the complex numbers again
  x = x.apply(std::conj);

  // scale the numbers
  x /= N;
}

const double& Math36::getFred(const double& sample, const double& nsample, const double& sampleFreq)
{
  return sampleFreq / nsample * sample;
}

void Math36::hilbert(CArray& x, const int& N)
{
  for (int i = 0; i < N; i++) {
    if (i % 2 == 0)
      x[i].imag(0.0);
    else
      x[i].imag(x[i].real() * 2.0 / M36_PI / double(i));
  }
}

void Math36::setSampleRate(const double& rate)
{
  sample_rate = rate;
}

Math36::QSO::QSO() : _n(0.0), _freq(0.0), _cos(0.0), _sin(0.0), _phase(0.0), _arg(0.0), _real_freq(1.0), _phase_reset(true)
{
  setFreq(0.0);
}

Math36::QSO::~QSO()
{
}

void Math36::QSO::next()
{
  _n += 1.0;
  _arg = _freq * _n + _phase;
  _sin = std::sin(_arg);
  _cos = std::cos(_arg);
}

void Math36::QSO::reset(const bool& preservePhase)
{
  if (preservePhase && !_phase_reset) {
    _phase_reset = false;
    _phase = _arg - double(int(double(_arg / 2.0 / M36_PI))) * 2.0 * M36_PI;
  }
  _n = -1.0;
  next();
}

double Math36::QSO::cos() const
{
	return _cos;
}

double Math36::QSO::sin() const
{
  return _sin;
}

void Math36::QSO::setFreq(const double& freq)
{
  if (freq != _real_freq) {
    _real_freq = freq;
    _phase = 0.0;
    _phase_reset = true;
    if (sample_rate != 0.0) {
      _freq = 2.0 * M36_PI * freq / sample_rate;
    }
    else _freq = freq;
  }
}

double Math36::QSO::getFreq() const
{
  return _real_freq;
}
