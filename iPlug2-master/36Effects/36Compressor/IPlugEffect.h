// #pragma once

#include "IPlug_include_in_plug_hdr.h"
#include "IControls.h"
#include "config.h"
#include <unordered_map>
#include <mutex>
// #include "Common.h"

const int kNumPrograms = 1;
const double soundSpeed = 3430.0; // cm per sec
const int maxBuffSize = 16384; // 800;
// const int maxBuffSize = 2 * PLUG_LATENCY;
const int displayLoop = 50;

const bool testPlugin = false;



enum EParams
{
  dcR = 0,
  limiter,
  limiterType,
  threshold,
  maxTresh,
  cp0,
  cp1,
  cp2,
  cp3,
  cp4,
  outgain,
  steep,
  sendDiff,
  replot,
  mix,
  kNumParams
};

enum EControlTags
{
  kCtrlTagLim = 0,
  kCtrlTagOutput,
  kNumCtrlTags
};

using namespace iplug;
using namespace igraphics;

class Limiter {
public:
  enum LType {
    MIRROR = 0,
    TRUNC
  };
  Limiter();
  ~Limiter();

  void ProcessBlock(sample** toProcess, const sample & thresh, const int& nFrames, const int& from, const int& buffSize);
  void ProcessBlock(sample toProcess[2][maxBuffSize], const sample& thresh, const int& nFrames, const int& from, const int& buffSize);
  sample processSample(sample& smp, const sample& thresh);

  void setChannels(const int& n);
  sample algo(const sample& input, const sample &thresh);
  void setType(const LType& ltype);
  void setPeakSender(IPeakSender<2>* psender, const int &ctrlTag);

private:
  LType type;
  int nChans, cTag;
  sample** diff;
  IPeakSender<2>* sender;
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
  void setSteepness(const double& steepness);

private:
  double steep, c, v;
  SType type;
};



class IPlugEffect final : public Plugin
{
public:
  IPlugEffect(const InstanceInfo& info);

#if IPLUG_DSP // http://bit.ly/2S64BDd
  void ProcessBlock(sample** inputs, sample** outputs, int nFrames) override;
  const void updateDisplay(double& var, EParams param, bool &update);
  void dcBlock(sample** inputs);
  void OnIdle() override;
  void OnUIClose() override;
  void OnUIOpen() override;
  // void OnActivate();
  //  void OnReset() override;
  sample getValue(const sample& s, const int& side);
#endif
private:

  void testPlug();
  void updateCurve();

  IPeakSender<2> mOutSender, mLimSender;

  sample y_buffer[2][maxBuffSize], input_cpy[2][maxBuffSize], process_buffer[2][maxBuffSize], out_buffer[2][maxBuffSize], last_input[2], last_delta[2], last_buffer[2];
  int ldot[2], ldot_age[2], send_pos, pro_pos;
  double mdcR, var[2], lvar[2], tresh,
    mtreshp, mtresh, msteepness, mmix;
  bool UIClosed, isInit, bsendDiff, breplot;
  int displayCount, atStartCount;

  Limiter limiter;

  double prev_stat[2][10][displayLoop];

public:
  double my_stat[2][10] = { {0.0} },
    cur_stat[2][10] = { {0.0} },
    _cp0,
    _cp1,
    _cp2,
    _cp3,
    _cp4;

  Sigmoid sigmoid;


};
