// #pragma once

#include "IPlug_include_in_plug_hdr.h"
#include "IControls.h"
#include "config.h"
// #include "Common.h"

const int kNumPrograms = 1;
const double soundSpeed = 3430.0; // cm per sec
const int maxBuffSize = 16384; // 800;
// const int maxBuffSize = 2 * PLUG_LATENCY;

const bool testPlugin = false;

enum EParams
{
  dfreq = 0,
  type,
  kNumParams
};

using namespace iplug;
using namespace igraphics;


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
  sample getValue(const sample& s, const int& side) const;
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
};
