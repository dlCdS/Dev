#pragma once
#include <complex>
#include <valarray>
#include "IPlugConstants.h"

typedef std::complex<double> Complex;
typedef std::valarray<Complex> CArray;

struct PolarComplex {
  double norm, arg;
};

typedef std::valarray<PolarComplex> PArray;

namespace Math36 {

  static const double M36_PI = 3.1415926535897932384626433832795;

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
  
  class Filter {
  public:
    void setPole(const Complex& pole);
    virtual double process(const double& x) = 0;
  protected:
    Complex _pole;
    double _pnorm;
  };


  class AllpassFilter : public Filter {
  public:
    AllpassFilter();

    virtual double process(const double& x);

  protected:
    double _x1, _x2, _y1, _y2;
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
  
}