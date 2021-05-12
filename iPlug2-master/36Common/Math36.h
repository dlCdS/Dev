#pragma once
#include <complex>
#include <valarray>
#include "IPlugConstants.h"
#define _USE_MATH_DEFINES
#include <math.h>
#include <vector>

typedef std::complex<double> Complex;
typedef std::valarray<Complex> CArray;

struct PolarComplex {
  double norm, arg;
};

typedef std::valarray<PolarComplex> PArray;

namespace Shape {
  enum Type {
    TRIANGLE = 0,
    SQUARE,
    SAWUP,
    SAWDOWN,
    SINE
  };

  double Sine(const double& t);
  double Triangle(const double& t);
  double Square(const double& t, const double& size = 0.5);
  double SawUp(const double& t);
  double SawDown(const double& t);
}

namespace Math36 {

  static const double M36_PI = 3.1415926535897932384626433832795;
  static const double EPSILON = 0.00001;

  class FFT {
  public:
    FFT();

    void setSize(const int& size);
    void setSampleRate(const double& rate);
    void feed(const double& val);
    void compute();
    double getFromFreq(const double& freq);
    double getFromIndex(const int& index);
  private:
    CArray _resarray, _array;
    int _size, _count;
    double _rate, _damp;
    std::vector<double> _display;
  };
  void fft(CArray& x, const int& N);


  void ifft(CArray& x, const int& N);

  const double& getFred(const double& sample, const double& nsample, const double& sampleFreq);

  void hilbert(CArray& x, const int& N);

  void setSampleRate(const double& rate);

  namespace {
    double sample_rate;
  }

  class Derivative {
  public:
    Derivative();
    ~Derivative();

    double get() const;
    double get(const double& next_value);
    void next(const double& next_value);
    bool setTimestep(const double& time_step);

    static double Derive(const double& x1, const double& x2, const double& timestep);

  protected:
    double last, value, timestep;
    bool first_use;
  };

  class Integral {
  public:
    enum Method {
      TRAP = 0
    };
    Integral();
    ~Integral();

    double get() const;
    double get(const double& next_value);
    virtual void next(const double& next_value);
    bool setTimestep(const double& time_step);

  protected:
    double last, value, timestep;
    bool first_use;


  };

  class PID {
  public:
    PID();
    ~PID();

    double get() const;
    double get(const double& next_value, const double& feedback = 0.0);
    virtual void next(const double& next_value, const double& feedback=0.0);
    bool setTimestep(const double& time_step);
    bool setPID(const double& p, const double& i, const double& d);

  protected:
    double last, value;

    double P, I, D;
    Integral xint;
    Derivative xder;
  };

  class AudioDb {
  public:
    AudioDb();
    ~AudioDb();

    bool setRatio(const double& ratio);
    double get() const;
    double get(const double& next_value);
    virtual void next(const double& next_value);

  protected:
    double value, ratio, ponder, db, xn_1, yn_1;
  };


  class AllpassFilter {
  public:
    AllpassFilter();

    void setPole(const Complex& pole);
    double process(const double& x);

  protected:
    Complex _pole;
    double _pnorm;
    double _x1, _x2, _y1, _y2;
  };

  class Filter {
  public:
    enum FilterMode {
      FILTER_MODE_LOWPASS = 0,
      FILTER_MODE_HIGHPASS,
      FILTER_MODE_BANDPASS,
      kNumFilterModes
    };

    Filter();

    double process(const double& inputValue);
    void setCutoff(const double& newCutoff, const double& samplerate);
    void setResonance(const double& newResonance);
    void setFilterMode(FilterMode newMode);
    void setAttenuation(const int& att);
    double getResponse(const double& freq) const;

  private:
    double cutoff, realcutoff;
    double resonance;
    double att;
    FilterMode mode;
    double feedbackAmount;
    void calculateFeedbackAmount();
    double buf0;
    double buf1;
    double buf2;
    double buf3;
    double* usedVal;
  };

  class QSO {
  public:
    QSO();
    ~QSO();

    void next();
    void reset(const bool& preservePhase=true);
    double cos() const;
    double sin() const;
    void setFreq(const double& freq);
    double getFreq() const;

  protected:
    double _n, _real_freq, _freq, _cos, _sin, _phase, _arg;
    bool _phase_reset;
  };

  class Sigmoid {
  public:
    enum SType {
      SIG = 0,
      LIN,
      REV
    };

    Sigmoid();
    ~Sigmoid();

    double get(const double& x) const;
    double rev(const double& x) const;
    double sig(const double& x) const;
    double linear(const double& x) const;
    bool setSteepness(const double& steepness);

  private:
    double steep, c, v;
    SType type;
  };

  const double TempoDivisonToDouble[] =
  {
    0.0625,   // 1 sixty fourth of a beat
    0.125,       // 1 thirty second of a beat
    0.166666,      // 1 sixteenth note tripet
    0.25,       // 1 sixteenth note
    0.333333,      // 1 dotted sixteenth note
    0.666666,        // 1 eigth note      // Corrected mess arounds definition and plugin display
    0.5,       // 1 dotted eigth note       // Corrected mess arounds definition and plugin display
    0.625,       // 1 eigth note tripet        // Corrected mess arounds definition and plugin display
    1.0,        // 1 quater note a.k.a 1 beat @ 4/4
    1.5,       // 1 dotted beat @ 4/4
    2.0,        // 2 beats @ 4/4
    4.0,          // 1 bar @ 4/4
    8.0,          // 2 bars @ 4/4
    16.0,          // 4 bars @ 4/4
    32.0          // 8 bars @ 4/4
  };

  const int TempoDivisonToDoubleSize = sizeof(TempoDivisonToDouble) / sizeof(double);
  class LFO {
  public:
    LFO();
    double get(const double& samplePerBeat, const int& samplePos);
    double get(const double& time);
    double getFromLocalTime();
    void increment(const double& time);
    void reset();
    void setMode(const bool& sync = true);
    void setFreq(const double& freq);
    void setRate(const int& type, const double& bpm);
    double getLastValue() const;
    void setType(const Shape::Type type);
    void setPhase(const double& phase);

  private:
    double getValue(const double& t);
    Shape::Type _type;
    double _time;
    double _period;
    double _rate;
    double _phase;
    bool _sync;
    double _last;
  };
}


