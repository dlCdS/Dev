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

void Math36::AllpassFilter::setPole(const Complex& pole)
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

Math36::Derivative::Derivative() : last(0.0), value(0.0), timestep(1.0), first_use(true)
{
}

Math36::Derivative::~Derivative()
{
}

double Math36::Derivative::get() const
{
  return value;
}

double Math36::Derivative::get(const double& next_value)
{
  next(next_value);
  return value;
}

void Math36::Derivative::next(const double& next_value)
{
  if (first_use) {
    last = next_value;
    value = 0.0;
    first_use = false;
  }
  else {
    value = (next_value - last) / timestep;
    last = next_value;
  }
}

bool Math36::Derivative::setTimestep(const double& time_step)
{
  if(time_step != 0.0) {
    timestep = time_step;
    return true;
  }
  return false;
}

double Math36::Derivative::Derive(const double& x1, const double& x2, const double& timestep)
{
  if (timestep != 0.0) {
    return (x2 - x1) / timestep;
  }
  return 0.0;
}

Math36::Integral::Integral() : last(0.0), value(0.0), timestep(1.0), first_use(true)
{
}

Math36::Integral::~Integral()
{
}

double Math36::Integral::get() const
{
  return value;;
}

double Math36::Integral::get(const double& next_value)
{
  next(next_value);
  return value;
}

void Math36::Integral::next(const double& next_value)
{
  if (first_use) {
    last = next_value;
    value = 0.0;
    first_use = false;
  }
  else {
    value = (next_value + last) / 2.0 * timestep;
    last = next_value;
  }
}

bool Math36::Integral::setTimestep(const double& time_step)
{
  if (time_step != 0.0) {
    timestep = time_step;
    return true;
  }
  return false;
}

Math36::PID::PID() : last(0.0), value(0.0), P(.5), I(.8), D(0.2)
{
}

Math36::PID::~PID()
{
}

double Math36::PID::get() const
{
  return value;
}

double Math36::PID::get(const double& next_value, const double& feedback)
{
  next(next_value, feedback);
  return value;
}

void Math36::PID::next(const double& next_value, const double& feedback)
{
  double loc(next_value - feedback);
  xint.next(loc);
  xder.next(loc);
  value = P *( loc + xint.get() / I + D * xder.get());
}

bool Math36::PID::setTimestep(const double& time_step)
{
  if (xder.setTimestep(time_step))
    if (xint.setTimestep(time_step))
      return true;
  return false;
}

bool Math36::PID::setPID(const double& p, const double& i, const double& d)
{
  P = p;
  D = d;
  if (i != 0.0) I = i;
  else return false;
  return true;
}

Math36::AudioDb::AudioDb() : value(0.0), ratio(0.0), ponder(0.0), db(0.0), xn_1(0.0), yn_1(0.0)
{
  setRatio(0.9);
}

Math36::AudioDb::~AudioDb()
{
}

bool Math36::AudioDb::setRatio(const double& new_ratio)
{
  if (ratio >= 0.0 && ratio < 1.0) {
    ratio = new_ratio;
    ponder = 1.0 / (1.0 - ratio);
    return true;
  }
  return false;
}

double Math36::AudioDb::get() const
{
  return db;
}

double Math36::AudioDb::get(const double& next_value)
{
  next(next_value);
  return get();
}

void Math36::AudioDb::next(const double& next_value)
{
  /// DC block for 0 centering
  yn_1 = next_value - xn_1 + 0.999 * yn_1;
  xn_1 = next_value;
  value = abs(yn_1) + value * ratio;
  db = 20 * log10(value/2.0);
}

Math36::Sigmoid::Sigmoid() : steep(0.0), c(1.0), v(1.0), type(SIG)
{
  setSteepness(1.0);
}

Math36::Sigmoid::~Sigmoid()
{
}

double Math36::Sigmoid::get(const double& x) const
{
  switch (type)
  {
  case Sigmoid::SIG:
    return sig(x);
    break;
  case Sigmoid::LIN:
    return linear(x);
    break;
  case Sigmoid::REV:
    return rev(x);
    break;
  default:
    return linear(x);
    break;
  }
}

double Math36::Sigmoid::rev(const double& x) const
{
  return -log(c / (v + x) - 1.0) / steep + 0.5;
}

double Math36::Sigmoid::sig(const double& x) const
{
  return c / (1.0 + exp(-steep * (x - 0.5))) - v;
}

double Math36::Sigmoid::linear(const double& x) const
{
  return x;
}

bool Math36::Sigmoid::setSteepness(const double& steepness)
{
  if (steep == steepness)
    return false;
  if (steepness < 0.0) {
    type = SType::REV;
    steep = -steepness;
    v = (1.0 + exp(-steep / 2.0)) / (exp(steep / 2) - exp(-steep / 2.0));
    c = v * (1.0 + exp(steep / 2.0));
  }
  else if (steepness > 0.0) {
    type = SType::SIG;
    steep = steepness;
    v = (1.0 + exp(-steep / 2.0)) / (exp(steep / 2) - exp(-steep / 2.0));
    c = v * (1.0 + exp(steep / 2.0));
  }
  else {
    type = SType::LIN;
  }
  return true;
}

Math36::Filter::Filter() : cutoff(0.1),
att(1.0),
realcutoff(log2(400)),
resonance(.9),
mode(Math36::Filter::FILTER_MODE_LOWPASS),
buf0(0.0),
buf1(0.0),
buf2(0.0),
buf3(0.0),
usedVal(&buf3)
{
  calculateFeedbackAmount();
}

double Math36::Filter::process(const double& inputValue)
{
  buf0 += cutoff * (inputValue - buf0 + feedbackAmount * (buf0 - buf1));
  buf1 += cutoff * (buf0 - buf1);
  buf2 += cutoff * (buf1 - buf2);
  buf3 += cutoff * (buf2 - buf3);
  switch (mode) {
  case FILTER_MODE_LOWPASS:
    return *usedVal;
  case FILTER_MODE_HIGHPASS:
    return inputValue - *usedVal;
  case FILTER_MODE_BANDPASS:
    return buf0 - *usedVal;
  default:
    return 0.0;
  }
}

void Math36::Filter::setCutoff(const double& newCutoff, const double& samplerate)
{
  realcutoff = log2(newCutoff);
  cutoff =  1.0 - exp(-1.0 / samplerate * 2.0 * M_PI * newCutoff);
  calculateFeedbackAmount();
}

void Math36::Filter::setResonance(const double& newResonance)
{
  resonance = newResonance;
  calculateFeedbackAmount();
}

void Math36::Filter::setFilterMode(FilterMode newMode)
{
  mode = newMode;
}

void Math36::Filter::setAttenuation(const int& attenuation)
{
  if (attenuation >= 0 && attenuation < 3){
    att = attenuation+1.0 ;
    switch (attenuation)
    {
    case 0:
      usedVal = &buf1;
      break;
    case 1:
      usedVal = &buf2;
      break;
    case 2:
      usedVal = &buf3;
      break;
    default:
      break;
    }
  }
    
}

double Math36::Filter::getResponse(const double& freq) const
{
  const double shaper = 1.;
  double locfreq = log2(freq);

  switch (mode)
  {
  case Math36::Filter::FILTER_MODE_LOWPASS:
    return 1. - 1. / exp(shaper * att * (-locfreq + realcutoff) + 1.) + resonance / (0.01*shaper * (-locfreq + realcutoff) * (-locfreq + realcutoff) + 1.0);
    break;
  case Math36::Filter::FILTER_MODE_HIGHPASS:
    return 1. - 1. / exp(shaper * att * (locfreq - realcutoff) + 1.) + resonance / (0.01*shaper *(-locfreq + realcutoff) * (-locfreq + realcutoff) + 1.0);
    break;
  case Math36::Filter::FILTER_MODE_BANDPASS:
    return 2. - 0.001*shaper * att * (locfreq - realcutoff)* (locfreq - realcutoff) + resonance / (0.01 * shaper * (-locfreq + realcutoff) * (-locfreq + realcutoff) + 1.0);
    break;
  case Math36::Filter::kNumFilterModes:
    break;
  default:
    break;
  }
}

void Math36::Filter::calculateFeedbackAmount()
{
  feedbackAmount = resonance + resonance / (1.0 - cutoff);
}

double Shape::Sine(const double& t)
{
	return sin(2.0 * M_PI * t);
}

double Shape::Triangle(const double& t)
{
  if (t < 0.5)
    return 4.0 * t - 1.0;
  else return  1.0 - 4.0 * (t - 0.5);
}

double Shape::Square(const double& t, const double& size)
{
  if (t < size)
    return 1.0;
  else return -1.0;
}

double Shape::SawUp(const double& t)
{
  return 2.0 * t - 1.0;
}

double Shape::SawDown(const double& t)
{
  return SawUp(1.0 - t);
}

Math36::LFO::LFO() : _type(Shape::Type::TRIANGLE),
_time(0.0),
_period(1.),
_rate(1.),
_phase(0.),
_sync(true)
{
}

double Math36::LFO::get(const double& samplePerBeat, const int& samplePos)
{
  double time = double(samplePos) / samplePerBeat;
  return getValue(time);
}

double Math36::LFO::get(const double& time)
{
  return getValue(time);
}

double Math36::LFO::getFromLocalTime()
{
  return getValue(_time);
}

void Math36::LFO::increment(const double& time)
{
  _time += time;
}

void Math36::LFO::reset()
{
  _time = 0.0;
}

void Math36::LFO::setMode(const bool& sync)
{
  _sync = sync;
}

void Math36::LFO::setFreq(const double& freq)
{
  if (freq > 0.0)
  _period = 1.0 / freq;
}

void Math36::LFO::setRate(const int& type, const double& bpm)
{if(type>=0 && type < TempoDivisonToDoubleSize)
  _rate = TempoDivisonToDouble[type] * bpm / 60.;
}

double Math36::LFO::getLastValue() const
{
  return _last;
}

void Math36::LFO::setType(const Shape::Type type)
{
  _type = type;
}

void Math36::LFO::setPhase(const double& phase)
{
  _phase = phase;
}

double Math36::LFO::getValue(const double &t)
{
  double loct;
  if (_sync) {
    loct = t / _rate;
  }
  else {
    loct = t / _period;
  }
  loct -= int(loct);
  loct += _phase;
  if (loct > 1.0)
    loct -= 1.0;
  if (loct > 1.0) loct -= 1.0;
  
  switch (_type)
  {
  case Shape::Type::SINE:
    _last = (1.0 + Shape::Sine(loct)) / 2.0;
    break;
  case Shape::Type::TRIANGLE:
    _last = (1.0 + Shape::Triangle(loct)) / 2.0;
    break;
  case Shape::Type::SQUARE:
    _last = (1.0 + Shape::Square(loct)) / 2.0;
    break;
  case Shape::Type::SAWUP:
    _last = (1.0 + Shape::SawUp(loct)) / 2.0;
    break;
  case Shape::Type::SAWDOWN:
    _last = (1.0 + Shape::SawDown(loct)) / 2.0;
    break;
  default:
    break;
  }
  return _last;
}

Math36::FFT::FFT() : _damp(0.9)
{
}

void Math36::FFT::setSize(const int& size)
{
  if(_size != size){
    _size = size;
    if (_array.size() != size) {
      _array.resize(size);
      _resarray.resize(size);
      _display.resize(size);
      for (int i = 0; i < _size; i++)
        _display[i] = 0.0;
      _count = 0;
    }
  }
}

void Math36::FFT::setSampleRate(const double& rate)
{
  _rate = rate;
}

void Math36::FFT::feed(const double& val)
{
  if(_count < _size)
    _array[_count] = Complex(val, 0.0);
  _count++;
}

void Math36::FFT::compute()
{
  double tmp;
  if(_count >= _size) {
    fft(_array, _size);
    for (int i = 0; i < _size / 2 + 1; i++){
      Complex sum = _array[i] + _array[_size - i];
      _display[i] = _damp * _display[i] + (sum.real() * sum.real() + sum.imag() * sum.imag()) / _size;
      if (_display[i] < 0.0 || _display[i] > 1000000.)_display[i] = 0.0;
    }
    _count = 0;
  }
}

double Math36::FFT::getFromFreq(const double& freq)
{
  double pos = freq * _size / _rate / 2., inter(pos - int(pos));
  int left(pos), right(pos + 1.0);

  if (left < 0) left = 0;
  else if (left >= _size) left = _size - 1;

  if (right < 0) right = 0;
  else if (right >= _size) right = _size - 1;

  return getFromIndex(left) * (1.0 - inter)
    + getFromIndex(right) * inter;
}

double Math36::FFT::getFromIndex(const int& index)
{
  if (index >= 0 && index < (_size) / 2 + 1){
    return log(1.0 + _display[index]);
  }
  else return 0.0;
}
